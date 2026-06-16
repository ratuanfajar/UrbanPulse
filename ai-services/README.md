# Slum Detection API (FastAPI + SHAP + LLM)

Serves the XGBoost slum classifier exported from the notebook (Section 13) with three layers:

1. **Prediction** — probability + label per kelurahan.
2. **Explainable AI** — local SHAP turned into plain-language Indonesian by an LLM.
3. **Live map pipeline** — click a point → resolve the kelurahan → run the real
   Google Earth Engine feature extraction (the same recipe as `04_extract_features_buildings.py`)
   → predict + explain. No "lookup from the training sheet".

## Project layout (run from inside this folder)

```
Hackathon-ML/
├── .venv/                       # the same venv you trained in
├── models/                      # produced by notebook Section 13
│   ├── slum_xgb_pipeline.joblib
│   └── model_metadata.json
├── data/processed/
│   └── labels.geojson           # kelurahan polygons (unit_id, city, slum, unit_area_m2)
└── ai-services/                 # THIS folder — run uvicorn here
    ├── app.py
    ├── model_service.py
    ├── pipeline_service.py      # point -> kelurahan -> GEE features
    ├── llm_service.py
    ├── feature_meta.py
    ├── schemas.py
    └── .env
```

## Install

```bash
uv add fastapi "uvicorn[standard]" pydantic python-dotenv joblib numpy pandas \
       scikit-learn xgboost shap openai earthengine-api geopandas shapely
```

Use the **same venv/versions** you trained with so `joblib` loads the model cleanly.

## Configure (`.env`)

```bash
cp .env.example .env
```

| Variable | Purpose |
|----------|---------|
| `MODELS_DIR` | folder with the exported model (default `../models`) |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `LLM_MODEL` | LLM for explanations (optional). For OpenRouter: base `https://openrouter.ai/api/v1`, model e.g. `openai/gpt-4o-mini` |
| `GEE_PROJECT` | Earth Engine project id (same as script 04). **Empty = live prediction for areas outside the local data is disabled** |
| `BOUNDARIES_PATH` | kelurahan polygons for the map + point lookup (default `../data/processed/labels.geojson`) |
| `FEATURES_CSV` | pre-extracted features for the training units (instant predictions, no GEE) |
| `RADIUS_M` | buffer radius (m) used when a click/search is **outside** every known kelurahan (default `500`) |
| `FEATURE_CACHE_DIR` | caches freshly extracted features so repeat clicks skip GEE |
| `PIPELINE_YEAR` | satellite composite year (must match training: `2023`) |

Authenticate Earth Engine once on this machine (same account as the notebook):

```bash
earthengine authenticate
```

## Run

```bash
uvicorn app:app --reload --port 8000
```

- **Map UI:** http://localhost:8000/ — click a point, press **Prediksi**, see the
  prediction, confidence vs threshold, top SHAP factors, and the Indonesian explanation,
  with the kelurahan highlighted on the map.
- API docs: http://localhost:8000/docs

The UI is a single static page (`static/index.html`, Leaflet via CDN) served by FastAPI
itself — no Node/build step. There are **three ways to pick an area**, none limited to a
fixed dropdown:

1. **Click a colored polygon** (training kelurahan) → instant prediction from the local
   cache, no GEE needed.
2. **Click anywhere else on the map** → drops a marker and predicts a `RADIUS_M` buffer
   around that point (needs `GEE_PROJECT`).
3. **Type a place name + Enter** → geocodes via `/geocode` (OpenStreetMap), recenters the
   map, and treats it as a point (needs `GEE_PROJECT`).

The boundary overlay loads from `/boundaries` and only needs `BOUNDARIES_PATH` — **no GEE
required**. Predicting any area *outside* the local cache (options 2 and 3) needs
`GEE_PROJECT` set, because fresh satellite features must be fetched for unseen areas.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | liveness, feature count, threshold, whether pipeline is enabled |
| GET | `/metadata` | threshold, required raw features, engineered formulas, validation metrics |
| POST | `/predict` | batch prediction from raw features |
| POST | `/explain` | one kelurahan from raw features: probability + SHAP + LLM |
| GET | `/predict-by-unit?unit_id=` | predict a known training unit by id — no GEE needed |
| **POST** | **`/predict-by-point`** | **map click/search `{lat,lon,radius_m?}` → cache or live GEE → predict + explain** |
| GET | `/areas?city=&q=` | list known training units (drives the search dropdown) — no GEE needed |
| GET | `/boundaries?city=` | kelurahan polygons (GeoJSON) for the map — no GEE needed |
| GET | `/geocode?q=` | geocode any place name to lat/lon (OpenStreetMap proxy) — no GEE needed |

### `/predict-by-point` (the map flow)

```bash
curl -X POST http://localhost:8000/predict-by-point \
  -H "Content-Type: application/json" \
  -d '{"lat":-0.502,"lon":117.153,"top_k":8}'
```

What happens:
1. Point-in-polygon checks the local cache first — if the click is inside a training
   kelurahan, its cached features are reused instantly (no GEE).
2. Otherwise, with `GEE_PROJECT` set: if `BOUNDARIES_PATH` covers the point, GEE extracts
   features **for that exact polygon**; if not, it extracts a **`RADIUS_M` buffer** around
   the point — so prediction works for **any** area, not just the training cities. Both use
   the script-04 recipe (Open Buildings + Sentinel-1/2 + 2.5D temporal) and are cached.
3. Compute the 7 engineered features → model → probability + local SHAP → LLM narrative.

Response includes `geometry` (highlight the area on the map), `feature_source`
(`"cache"`, `"gee"`, or `"gee_radius"`), `radius_m` (set when the buffer fallback was
used), and `ground_truth` (the KOTAKU label, only when a labeled polygon matched — shown
for comparison, **never fed to the model**). Pass an optional `"radius_m"` in the body to
override the default buffer size for a single request.

## Why is a boundaries file needed at all?

Prediction has **nothing to do with the training data** — every request fetches fresh
satellite data from GEE. The boundaries file exists for one reason: the model was trained
on **per-kelurahan aggregates** (building density, NDVI mean over the area, coverage ratio,
etc.), so inference needs an **area to aggregate over**, not a bare point. A click gives a
point; the boundaries file turns that point into the kelurahan polygon it falls inside.

It does **not** have to be `labels.geojson` — that's just the default because it conveniently
ships the polygons plus the slum label. Two ways to predict **anywhere in Indonesia**:

- **Best — a nationwide boundaries file:** point `BOUNDARIES_PATH` at a country-wide
  kelurahan file (e.g. GADM level-4 or geoBoundaries ADM4). Every click then resolves to a
  real administrative polygon, matching how the model was trained.
- **Out of the box — radius buffer:** if the click falls outside `BOUNDARIES_PATH`, the API
  aggregates over a circular `RADIUS_M` buffer around the point instead. This needs no extra
  file and works for any coordinate. Caveat: the model was trained on kelurahan-shaped
  footprints, so a fixed radius adds some train/serve skew — treat buffer results
  (`feature_source: "gee_radius"`) as indicative rather than authoritative, and tune
  `RADIUS_M` to the typical kelurahan size in your target area.

## Two honest notes about "leakage"

- **The pipeline is real, not a sheet lookup.** Features are re-extracted from GEE at
  request time using the training recipe, so there is no train/serve skew.
- **But generalization still depends on WHERE you click.** The 4 training cities' kelurahan
  are in `labels.geojson`; clicking inside them predicts on polygons the model saw during
  training, so results look optimistic. To demonstrate true generalization, point
  `BOUNDARIES_PATH` at kelurahan boundaries of a city **not** in training — the pipeline is
  identical, the area is just unseen. The honest headline metric remains the LOCO result
  (recall 0.91 @ threshold 0.185).

## Files

| File | Role |
|------|------|
| `app.py` | FastAPI app + endpoints (pipeline is lazy-loaded) |
| `model_service.py` | model loading, feature engineering, predict, local SHAP |
| `pipeline_service.py` | point → kelurahan → GEE feature extraction + cache |
| `boundaries_service.py` | loads kelurahan polygons + city list (geopandas only, no GEE) |
| `llm_service.py` | LLM explanation + deterministic fallback |
| `feature_meta.py` | Indonesian human-readable feature names |
| `schemas.py` | request models |

## Swapping the LLM provider

All LLM logic is isolated in `llm_service.py`. Change the client call in `explain_with_llm`
and keep the endpoint contract identical.
