import ee
import os
import time
import geopandas as gpd
import pandas as pd

# PROJECT = "elated-parser-499508-j7"
PROJECT = "exemplary-oven-352202"
LABELS  = "data/processed/labels.geojson"
OUT     = "data/processed/features_buildings.csv"
YEAR    = 2023
CHUNK   = 20
S2_START, S2_END = f"{YEAR-1}-01-01", f"{YEAR}-12-31"
S1_START, S1_END = f"{YEAR}-01-01", f"{YEAR}-12-31"

ee.Initialize(project=PROJECT)

OB = (ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")
      .filter(ee.Filter.gte("confidence", 0.65)))
AREA_RED = (ee.Reducer.count()
            .combine(ee.Reducer.sum(), sharedInputs=True)
            .combine(ee.Reducer.mean(), sharedInputs=True)
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
            .combine(ee.Reducer.median(), sharedInputs=True))
RAST_RED = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)


def s2_composite(region):
    csp = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterDate(S2_START, S2_END).filterBounds(region)
          .linkCollection(csp, ["cs"])
          .map(lambda i: i.updateMask(i.select("cs").gte(0.6))))
    img = s2.median().divide(10000)
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi")
    ndbi = img.normalizedDifference(["B11", "B8"]).rename("ndbi")
    bsi = img.expression(
        "((B11+B4)-(B8+B2))/((B11+B4)+(B8+B2))",
        {"B11": img.select("B11"), "B4": img.select("B4"),
         "B8": img.select("B8"), "B2": img.select("B2")}).rename("bsi")
    bright = img.select(["B2", "B3", "B4"]).reduce(ee.Reducer.mean()).rename("brightness")
    gray = ndbi.add(1).multiply(50).toInt()
    glcm = gray.glcmTexture(size=3).select(["ndbi_contrast", "ndbi_ent", "ndbi_var"])
    return ndvi.addBands([ndbi, bsi, bright, glcm])


def s1_composite(region):
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterDate(S1_START, S1_END).filterBounds(region)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"]))
    m = s1.median()
    return m.addBands(m.select("VV").subtract(m.select("VH")).rename("vv_vh"))


def temporal_2p5d(region):
    return (ee.ImageCollection("GOOGLE/Research/open-buildings-temporal/v1")
            .filterBounds(region)
            .filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31")
            .select(["building_fractional_count", "building_height", "building_presence"])
            .mosaic())


def to_fc(sub):
    sub = sub.copy()
    sub["geometry"] = sub.geometry.simplify(0.0002, preserve_topology=True)
    feats = []
    for _, r in sub.iterrows():
        g = ee.Geometry(r.geometry.__geo_interface__, proj="EPSG:4326", geodesic=False)
        feats.append(ee.Feature(g, {
            "unit_id": r.unit_id, "city": r.city,
            "slum": int(r.slum), "unit_area_m2": float(r.unit_area_m2)}))
    return ee.FeatureCollection(feats)


def make_feat(ob, raster):
    def feat(unit):
        g = unit.geometry()
        b = ob.filterBounds(g)
        bs = b.reduceColumns(AREA_RED, ["area_in_meters"])
        rs = raster.reduceRegion(reducer=RAST_RED, geometry=g, scale=10,
                                 maxPixels=1e9, bestEffort=True, tileScale=16)
        area = ee.Number(unit.get("unit_area_m2"))
        n = ee.Number(bs.get("count"))
        return unit.set(rs).set({
            "b_count": n,
            "b_area_sum": bs.get("sum"),
            "b_area_mean": bs.get("mean"),
            "b_area_std": bs.get("stdDev"),
            "b_area_median": bs.get("median"),
            "b_coverage": ee.Number(bs.get("sum")).divide(area),
            "b_density_km2": n.divide(area).multiply(1e6)})
    return feat


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
        feat = make_feat(OB.filterBounds(region),
                         s2_composite(region).addBands(s1_composite(region)).addBands(temporal_2p5d(region)))
        for i in range(0, len(sub), CHUNK):
            chunk = sub.iloc[i:i + CHUNK]
            rows.extend(get_chunk(chunk, feat))         
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"      {min(i + CHUNK, len(sub))}/{len(sub)}", flush=True)

    df = pd.DataFrame(rows)
    for c in ["b_count", "b_area_sum", "b_coverage", "b_density_km2"]:
        df[c] = df[c].fillna(0)
    df["b_size_cv"] = df["b_area_std"] / df["b_area_mean"]
    df.to_csv(OUT, index=False)
    print(f"\n  saved {len(df)} rows x {df.shape[1]} cols -> {OUT}")
    print(f"  slum={int(df.slum.sum())}  non-slum={int((df.slum == 0).sum())}")


if __name__ == "__main__":
    main()