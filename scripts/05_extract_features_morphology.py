import ee
import os
import math
import time
import geopandas as gpd
import pandas as pd

# PROJECT    = "elated-parser-499508-j7"
PROJECT = "exemplary-oven-352202"
LABELS     = "data/processed/labels.geojson"
OUT        = "data/processed/features_morphology.csv"
CHUNK      = 10
MASK_SCALE = 5

ee.Initialize(project=PROJECT)

OB = (ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")
      .filter(ee.Filter.gte("confidence", 0.65)))
COMP_RED = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
RAST_RED = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)


def to_fc(sub):
    sub = sub.copy()
    sub["geometry"] = (sub.geometry
                       .simplify(0.0002, preserve_topology=True)
                       .buffer(0))
    feats = []
    for _, r in sub.iterrows():
        g = ee.Geometry(r.geometry.__geo_interface__, proj="EPSG:4326", geodesic=False)
        feats.append(ee.Feature(g, {"unit_id": r.unit_id, "city": r.city}))
    return ee.FeatureCollection(feats)


def make_feat(ob, gap, glcm):
    def feat(unit):
        g = unit.geometry()

        def add_pp(f):
            a = ee.Number(f.get("area_in_meters"))
            p = f.geometry().perimeter(1)
            return f.set("pp", a.multiply(4 * math.pi).divide(p.pow(2)))

        comp = ob.filterBounds(g).map(add_pp).reduceColumns(COMP_RED, ["pp"])
        gs = gap.reduceRegion(reducer=RAST_RED, geometry=g, scale=MASK_SCALE,
                              maxPixels=1e9, bestEffort=True, tileScale=16)
        ts = glcm.reduceRegion(reducer=ee.Reducer.mean(), geometry=g, scale=MASK_SCALE,
                               maxPixels=1e9, bestEffort=True, tileScale=16)
        return unit.set({
            "g_compact_mean": comp.get("mean"),
            "g_compact_std": comp.get("stdDev"),
            "h_gap_mean": gs.get("distance_mean"),
            "h_gap_std": gs.get("distance_stdDev"),
            "h_layout_ent": ts.get("bld_ent"),
            "h_layout_contrast": ts.get("bld_contrast"),
        })
    return feat


def build_layers(region):
    ob_r = OB.filterBounds(region)
    bld = (ee.Image(0).byte().paint(ob_r, 1)
           .reproject(crs="EPSG:3857", scale=MASK_SCALE)
           .rename("bld"))
    gap = (bld.fastDistanceTransform(256).sqrt()
           .multiply(MASK_SCALE).updateMask(bld.eq(0)).rename("distance"))
    glcm = bld.glcmTexture(size=4).select(["bld_ent", "bld_contrast"])
    return ob_r, gap, glcm


def get_one(fc, feat, retries=4):
    for k in range(retries):
        try:
            return [f["properties"] for f in fc.map(feat).getInfo()["features"]]
        except Exception:
            if k == retries - 1:
                return None
            time.sleep(8)


def get_chunk(sub_chunk, feat, retries=3):
    fc = to_fc(sub_chunk)
    for k in range(retries):
        try:
            return [f["properties"] for f in fc.map(feat).getInfo()["features"]]
        except Exception as e:
            if k == retries - 1:
                break
            print(f"      retry {k+1} ({type(e).__name__})", flush=True)
            time.sleep(8)
    print("      fallback per-unit", flush=True)
    out = []
    for idx, r in sub_chunk.iterrows():
        res = get_one(to_fc(sub_chunk.loc[[idx]]), feat)
        if res is None:
            print(f"      SKIP unit {r.unit_id}", flush=True)
        else:
            out.extend(res)
    return out


def main():
    gdf = gpd.read_file(LABELS).to_crs(4326)
    rows, done = [], set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        rows = prev.to_dict("records")
        done = set(zip(prev.city, prev.unit_id))
        print(f"  resume: {len(done)} unit sudah ada", flush=True)

    for city in sorted(gdf.city.unique()):
        sub = gdf[gdf.city == city]
        sub = sub[~sub.apply(lambda r: (r.city, r.unit_id) in done, axis=1)]
        if len(sub) == 0:
            print(f"  {city:10s} | selesai (skip)", flush=True)
            continue
        print(f"  {city:10s} | sisa={len(sub)}", flush=True)
        region = to_fc(sub).geometry().bounds()
        ob_r, gap, glcm = build_layers(region)
        feat = make_feat(ob_r, gap, glcm)
        for i in range(0, len(sub), CHUNK):
            chunk = sub.iloc[i:i + CHUNK]
            rows.extend(get_chunk(chunk, feat))
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"      {min(i + CHUNK, len(sub))}/{len(sub)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\n  saved {len(df)} rows x {df.shape[1]} cols -> {OUT}")


if __name__ == "__main__":
    main()