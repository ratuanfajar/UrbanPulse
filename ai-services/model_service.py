"""Model loading, feature engineering, prediction, and local SHAP explanations.

The feature-engineering block here MUST stay byte-for-byte equivalent to
Section 2.5 of the training notebook, otherwise inference will silently drift.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from feature_meta import FEATURE_DESCRIPTIONS

EPS = 1e-6


class ModelService:
    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)

        with open(self.models_dir / "model_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.pipeline = joblib.load(self.models_dir / self.metadata["model_file"])
        self.feature_order: list[str] = list(self.metadata["feature_order"])
        self.raw_features: list[str] = list(self.metadata["raw_features_required"])
        self.engineered: list[str] = list(self.metadata["engineered_features"])
        # Operating threshold comes from the exported metadata (chosen in the
        # notebook's threshold-comparison section). It can be overridden at
        # runtime with SLUM_THRESHOLD for quick A/B testing without re-exporting.
        self.threshold: float = float(self.metadata["operating_threshold"])
        _thr_env = os.getenv("SLUM_THRESHOLD")
        if _thr_env:
            try:
                self.threshold = float(_thr_env)
            except ValueError:
                pass
        self.target: dict = self.metadata.get("target", {"0": "Non-Slum", "1": "Slum"})

        # Pipeline = SimpleImputer (median) -> XGBClassifier.
        self._imputer = self.pipeline.named_steps["imputer"]
        self._clf = self.pipeline.named_steps["clf"]
        # Rebuild the explainer from the loaded model (more portable than a pickled one).
        self._explainer = shap.TreeExplainer(self._clf)

    # ---- Feature engineering: must mirror notebook Section 2.5 exactly ----
    def _add_engineered(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def col(name: str) -> pd.Series:
            return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

        df["b_area_med_mean_ratio"] = col("b_area_median") / (col("b_area_mean") + EPS)
        df["b_coverage_per_count"] = col("b_coverage") / (col("b_count") + EPS)
        df["density_lowrise"] = col("b_density_km2") / (col("building_height_mean") + EPS)
        df["coverage_per_area"] = col("b_coverage") / (col("b_area_mean") + EPS)
        df["ndbi_minus_ndvi"] = col("ndbi_mean") - col("ndvi_mean")
        df["ndbi_texture_ratio"] = col("ndbi_contrast_mean") / (col("ndbi_mean").abs() + EPS)
        df["vh_vv_diff"] = col("VH_mean") - col("VV_mean")

        # Divisions can produce inf -> NaN so the median imputer handles them.
        df[self.engineered] = df[self.engineered].replace([np.inf, -np.inf], np.nan)
        return df

    def _prepare(self, records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        df = self._add_engineered(df)
        # Guarantee every model column exists in the exact training order.
        for f in self.feature_order:
            if f not in df.columns:
                df[f] = np.nan
        return df[self.feature_order].apply(pd.to_numeric, errors="coerce")

    def predict(self, records: list[dict]) -> list[dict]:
        X = self._prepare(records)
        proba = self.pipeline.predict_proba(X)[:, 1]
        preds = (proba >= self.threshold).astype(int)
        return [
            {
                "slum_probability": round(float(p), 6),
                "slum_prediction": int(c),
                "label": self.target[str(int(c))],
                "threshold": self.threshold,
            }
            for p, c in zip(proba, preds)
        ]

    def explain(self, record: dict, top_k: int = 8) -> dict:
        X = self._prepare([record])
        proba = float(self.pipeline.predict_proba(X)[0, 1])
        pred = int(proba >= self.threshold)

        X_imp = pd.DataFrame(self._imputer.transform(X), columns=self.feature_order)
        sv = np.asarray(self._explainer.shap_values(X_imp))
        if sv.ndim == 3:  # multiclass-shaped output -> take positive class
            sv = sv[:, :, -1]
        row = sv[0]

        contribs = [
            {"feature": f, "value": float(v), "shap_value": float(s)}
            for f, v, s in zip(self.feature_order, X_imp.iloc[0].values, row)
        ]
        contribs.sort(key=lambda d: abs(d["shap_value"]), reverse=True)
        top = contribs[:top_k]
        for c in top:
            c["pushes_toward"] = self.target["1"] if c["shap_value"] > 0 else self.target["0"]
            c["description"] = FEATURE_DESCRIPTIONS.get(c["feature"], c["feature"])
            c["value"] = round(c["value"], 4)
            c["shap_value"] = round(c["shap_value"], 4)

        base = float(np.asarray(self._explainer.expected_value).ravel()[-1])
        return {
            "slum_probability": round(proba, 6),
            "slum_prediction": pred,
            "label": self.target[str(pred)],
            "threshold": self.threshold,
            "base_value": round(base, 4),
            "top_features": top,
        }
