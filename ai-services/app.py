"""Slum Detection API — prediction + explainable AI (SHAP + LLM).

Core endpoints work WITHOUT Google Earth Engine. Boundaries and feature lookup
are served from local GeoJSON + CSV cache. GEE is only used if configured.

Run:
    uvicorn app:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from shapely.geometry import Point, shape

import llm_service
from model_service import ModelService
from schemas import ExplainRequest, PointRequest, PredictRequest

from chatbot.routes import router as chatbot_router
from chatbot.rag_service import RAGService

load_dotenv(Path(__file__).parent / ".env")

MODELS_DIR = os.getenv("MODELS_DIR", str(Path(__file__).parent.parent / "models"))
STATIC_DIR = Path(__file__).parent / "static"
_BASE = Path(__file__).parent.parent  # E:\competition\Hackathon-ML

# Local data paths — always available, no GEE needed
LABELS_PATH = os.getenv("BOUNDARIES_PATH", str(_BASE / "data/processed/labels.geojson"))
FEATURES_CSV = os.getenv("FEATURES_CSV", str(_BASE / "data/processed/features_buildings.csv"))

# Live-pipeline config (optional, enables live GEE extraction for arbitrary points)
GEE_PROJECT = os.getenv("GEE_PROJECT", "")
FEATURE_CACHE_DIR = os.getenv("FEATURE_CACHE_DIR", str(Path(__file__).parent / "feature_cache"))
PIPELINE_YEAR = int(os.getenv("PIPELINE_YEAR", "2023"))
# Buffer radius (meters) used when a click falls OUTSIDE every known kelurahan,
# so prediction is NOT limited to the training areas (requires GEE_PROJECT).
RADIUS_M = float(os.getenv("RADIUS_M", "500"))

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Slum Detection API", version="2.0.0")

app.include_router(chatbot_router, prefix="/chatbot")
app.state.rag_service = RAGService()
app.state.rag_service.initialize()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

svc = ModelService(MODELS_DIR)

# ── Local data (loaded once at startup) ───────────────────────────────────────
_geojson: dict | None = None
_features_df: pd.DataFrame | None = None
_unit_index: dict[str, dict] = {}   # unit_id -> {features, ground_truth, geometry, city, ...}
_geo_index: list[dict] = []          # [{geometry_obj, unit_id, city, ...}] for spatial lookup


def _load_local_data():
    global _geojson, _features_df, _unit_index, _geo_index

    # Load GeoJSON boundaries
    labels_path = Path(LABELS_PATH)
    if not labels_path.exists():
        return
    with open(labels_path, encoding="utf-8") as f:
        _geojson = json.load(f)

    # Load feature CSV
    feat_path = Path(FEATURES_CSV)
    if feat_path.exists():
        _features_df = pd.read_csv(feat_path)

    # Build indexes
    feat_lookup: dict[str, dict] = {}
    if _features_df is not None and "unit_id" in _features_df.columns:
        for _, row in _features_df.iterrows():
            feat_lookup[str(row["unit_id"])] = row.to_dict()

    for feature in (_geojson or {}).get("features", []):
        props = feature.get("properties", {})
        geom_dict = feature.get("geometry")
        unit_id = str(props.get("unit_id", ""))
        city = str(props.get("city", ""))
        unit_name = str(props.get("unit_name") or unit_id)
        ground_truth = props.get("slum")
        kumuh_level = props.get("kumuh_level", "")
        unit_area = props.get("unit_area_m2")

        feat_row = feat_lookup.get(unit_id, {})
        entry = {
            "unit_id": unit_id,
            "unit_name": unit_name,
            "city": city,
            "ground_truth": int(ground_truth) if ground_truth is not None else None,
            "kumuh_level": kumuh_level,
            "unit_area_m2": unit_area,
            "geometry": geom_dict,
            "features": {k: v for k, v in feat_row.items()
                         if k not in ("unit_id", "city", "slum", "unit_area_m2")},
        }
        _unit_index[unit_id] = entry
        if geom_dict:
            try:
                _geo_index.append({"shape": shape(geom_dict), **entry})
            except Exception:
                pass


_load_local_data()

# ── Optional GEE pipeline ──────────────────────────────────────────────────────
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not GEE_PROJECT:
            raise HTTPException(
                status_code=503,
                detail="Live GEE pipeline disabled. Set GEE_PROJECT in .env to enable.",
            )
        try:
            from pipeline_service import PipelineService
            _pipeline = PipelineService(
                LABELS_PATH, GEE_PROJECT,
                year=PIPELINE_YEAR, cache_dir=FEATURE_CACHE_DIR,
            )
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Pipeline init failed: {e}")
    return _pipeline


# ── Helpers ────────────────────────────────────────────────────────────────────
def _find_unit_by_point(lat: float, lon: float) -> dict | None:
    """Spatial lookup: which unit polygon contains the clicked point."""
    pt = Point(lon, lat)
    for entry in _geo_index:
        try:
            if entry["shape"].covers(pt):
                return entry
        except Exception:
            pass
    return None


def _explain_unit(entry: dict, top_k: int = 8) -> dict:
    features = entry.get("features", {})
    expl = svc.explain(features, top_k=top_k)
    narrative = llm_service.explain_with_llm(
        expl, unit_id=entry.get("unit_id"), city=entry.get("city"))
    return {
        "unit_id": entry.get("unit_id"),
        "unit_name": entry.get("unit_name"),
        "city": entry.get("city"),
        "ground_truth": entry.get("ground_truth"),
        "kumuh_level": entry.get("kumuh_level"),
        "geometry": entry.get("geometry"),
        "feature_source": "cache" if entry.get("features") else "imputed",
        **expl,
        **narrative,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({
        "message": "Slum Detection API v2",
        "endpoints": ["/health", "/metadata", "/boundaries", "/areas",
                      "/predict", "/explain", "/predict-by-unit",
                      "/predict-by-point", "/geocode"],
        "docs": "/docs",
    })


@app.get("/health")
def health():
    return {
        "status": "ok",
        "n_features": len(svc.feature_order),
        "threshold": svc.threshold,
        "n_units_loaded": len(_unit_index),
        "gee_enabled": bool(GEE_PROJECT),
    }


@app.get("/metadata")
def metadata():
    keys = ("operating_threshold", "raw_features_required", "engineered_features",
            "engineered_formulas", "validation", "target")
    return {k: svc.metadata[k] for k in keys if k in svc.metadata}


@app.get("/boundaries")
def boundaries(city: str | None = None):
    """GeoJSON of all kelurahan polygons — served from local file, no GEE needed."""
    if _geojson is None:
        raise HTTPException(status_code=503, detail="Boundary data not loaded.")
    if city:
        filtered = [f for f in _geojson["features"]
                    if str(f.get("properties", {}).get("city", "")).lower() == city.lower()]
        return {**_geojson, "features": filtered}
    return _geojson


@app.get("/areas")
def areas(city: str | None = None, q: str | None = None):
    """List all known units (for dropdown/search). Optionally filter by city or name query."""
    results = []
    for uid, entry in _unit_index.items():
        if city and entry.get("city", "").lower() != city.lower():
            continue
        if q:
            q_lower = q.lower()
            if q_lower not in entry.get("unit_name", "").lower() \
               and q_lower not in uid.lower() \
               and q_lower not in entry.get("city", "").lower():
                continue
        results.append({
            "unit_id": uid,
            "unit_name": entry.get("unit_name"),
            "city": entry.get("city"),
            "ground_truth": entry.get("ground_truth"),
            "kumuh_level": entry.get("kumuh_level"),
        })
    results.sort(key=lambda x: (x.get("city") or "", x.get("unit_name") or ""))
    return {"areas": results, "total": len(results)}


@app.post("/predict")
def predict(req: PredictRequest):
    records = [r.model_dump(exclude_none=True) for r in req.records]
    results = svc.predict(records)
    for rec, res in zip(records, results):
        res["unit_id"] = rec.get("unit_id")
        res["city"] = rec.get("city")
    return {"results": results}


@app.post("/explain")
def explain(req: ExplainRequest):
    data = req.model_dump(exclude_none=True)
    top_k = int(data.pop("top_k", 8))
    unit_id = data.pop("unit_id", None)
    city = data.pop("city", None)
    if not data:
        raise HTTPException(status_code=400, detail="No feature values provided.")
    expl = svc.explain(data, top_k=top_k)
    narrative = llm_service.explain_with_llm(expl, unit_id=unit_id, city=city)
    return {"unit_id": unit_id, "city": city, **expl, **narrative}


@app.get("/predict-by-unit")
def predict_by_unit(unit_id: str, top_k: int = 8):
    """Predict and explain a known unit by its unit_id (no GEE needed)."""
    entry = _unit_index.get(unit_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unit '{unit_id}' not found.")
    return _explain_unit(entry, top_k=top_k)


@app.post("/predict-by-point")
def predict_by_point(req: PointRequest):
    """Map click -> predict ANY point in Indonesia.

    1) If the click is inside a cached kelurahan, use it (instant, no GEE).
    2) Otherwise, if GEE is enabled, extract features live: the exact kelurahan
       polygon when the click falls inside one, else a radius buffer around the
       point — so detection is NOT limited to the training areas.
    """
    # First try local spatial lookup (training kelurahan, no GEE needed).
    entry = _find_unit_by_point(req.lat, req.lon)
    if entry is not None:
        return _explain_unit(entry, top_k=req.top_k)

    if not GEE_PROJECT:
        raise HTTPException(
            status_code=503,
            detail="This area is outside the local dataset. Enable GEE_PROJECT in .env "
                   "to predict any location in Indonesia.",
        )

    pipe = get_pipeline()
    radius = req.radius_m or RADIUS_M
    # Exact polygon if the boundaries file covers this point, else a radius buffer.
    bundle = pipe.features_for_point(req.lat, req.lon)
    if bundle is None:
        bundle = pipe.features_for_buffer(req.lat, req.lon, radius_m=radius)

    expl = svc.explain(bundle["features"], top_k=req.top_k)
    narrative = llm_service.explain_with_llm(
        expl, unit_id=bundle["unit_id"], city=bundle["city"])
    return {
        "unit_id": bundle["unit_id"],
        "unit_name": bundle.get("unit_id"),
        "city": bundle["city"],
        "ground_truth": bundle["ground_truth"],
        "feature_source": bundle["feature_source"],
        "radius_m": radius if bundle["feature_source"] == "gee_radius" else None,
        "geometry": bundle["geometry"],
        **expl,
        **narrative,
    }


@app.get("/geocode")
def geocode(q: str):
    """Proxy Nominatim geocoding so the frontend doesn't need CORS workaround."""
    import urllib.request
    import urllib.parse
    url = ("https://nominatim.openstreetmap.org/search?"
           + urllib.parse.urlencode({"q": q, "format": "json", "limit": 5}))
    req = urllib.request.Request(url, headers={"User-Agent": "SlumDetectionApp/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        return {"results": [{"display_name": r["display_name"],
                              "lat": float(r["lat"]),
                              "lon": float(r["lon"])} for r in data]}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Geocoding failed: {e}")

