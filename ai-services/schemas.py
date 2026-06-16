"""Pydantic request models. Raw feature values are accepted as extra fields."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class FeatureRecord(BaseModel):
    """One kelurahan. Send raw feature columns as fields; missing ones are imputed."""

    model_config = ConfigDict(extra="allow")
    unit_id: Optional[str] = None
    city: Optional[str] = None


class PredictRequest(BaseModel):
    records: list[FeatureRecord] = Field(..., min_length=1)


class ExplainRequest(BaseModel):
    """Single kelurahan to predict + explain. Raw features go as extra fields."""

    model_config = ConfigDict(extra="allow")
    unit_id: Optional[str] = None
    city: Optional[str] = None
    top_k: int = Field(default=8, ge=1, le=41)


class PointRequest(BaseModel):
    """A map click. Backend resolves the kelurahan and runs the live GEE pipeline."""

    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    top_k: int = Field(default=8, ge=1, le=41)
    # Radius (m) of the buffer used when the point is outside any known kelurahan.
    # None -> server default (RADIUS_M). Only used by the live GEE pipeline.
    radius_m: Optional[float] = Field(default=None, ge=50, le=5000)
