"""Download polygon RW Kumuh DKI Jakarta dari KOTAKU WebMap.
Output: dataset/Indonesia/labels/dki_kumuh.geojson + .shp
Usage: python scripts/download_kotaku.py
"""
import json
import sys
from pathlib import Path

import requests
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

# ----- Config -----
ITEM_ID = "1aa1482cb1b14afa8993ffa7d303f25c"
LAYER_TITLE = "Sebaran RW Kumuh"  # atau "Status RW Kumuh" (isinya identik)

OUT_DIR = Path(__file__).parent.parent / "dataset" / "Indonesia" / "labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def arcgis_polygon_to_shapely(arcgis_geom):
    """Konversi ArcGIS polygon (rings) → Shapely Polygon/MultiPolygon.
    ArcGIS: outer ring clockwise, holes counter-clockwise.
    Untuk simple case (no holes), treat semua rings sebagai polygon terpisah → MultiPolygon.
    """
    rings = arcgis_geom.get("rings", [])
    if not rings:
        return None
    polys = [Polygon(r) for r in rings if len(r) >= 4]
    if len(polys) == 1:
        return polys[0]
    return MultiPolygon(polys)


def main():
    print(f"[1/4] Download WebMap data dari ArcGIS Online...")
    url = f"https://www.arcgis.com/sharing/rest/content/items/{ITEM_ID}/data?f=pjson"
    data = requests.get(url, timeout=30).json()

    # Cari layer
    target_layer = None
    for layer in data["operationalLayers"]:
        if layer.get("title") == LAYER_TITLE:
            target_layer = layer
            break
    if target_layer is None:
        print(f"⚠ Layer '{LAYER_TITLE}' tidak ditemukan. Layer tersedia:")
        for l in data["operationalLayers"]:
            print(f"  - {l.get('title')}")
        sys.exit(1)

    inner_layer = target_layer["featureCollection"]["layers"][0]
    feature_set = inner_layer["featureSet"]
    features_raw = feature_set["features"]

    # spatialReference ada di layerDefinition atau di geometry tiap feature
    sr_obj = (
        inner_layer.get("layerDefinition", {}).get("spatialReference")
        or features_raw[0].get("geometry", {}).get("spatialReference")
        or {"wkid": 102100}
    )
    spatial_ref = sr_obj.get("latestWkid") or sr_obj.get("wkid")
    print(f"[2/4] Extract {len(features_raw)} features. Spatial ref: EPSG:{spatial_ref}")

    # Build GeoDataFrame
    rows = []
    for f in features_raw:
        attrs = f["attributes"].copy()
        geom = arcgis_polygon_to_shapely(f["geometry"])
        if geom is None:
            continue
        attrs["geometry"] = geom
        rows.append(attrs)

    # ArcGIS pakai EPSG:102100 (Esri code) yang setara EPSG:3857 (Web Mercator)
    src_crs = "EPSG:3857" if spatial_ref in (102100, 3857) else f"EPSG:{spatial_ref}"
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=src_crs)
    print(f"[3/4] Built GeoDataFrame: {len(gdf)} rows. CRS: {gdf.crs}")

    # Reproject ke WGS84 (lat/lon) untuk kompatibilitas dengan Sentinel-2 GEE export
    gdf_wgs = gdf.to_crs("EPSG:4326")

    # Save GeoJSON (WGS84) — untuk pipeline & visualisasi
    geojson_path = OUT_DIR / "dki_kumuh.geojson"
    gdf_wgs.to_file(geojson_path, driver="GeoJSON")
    print(f"[4/4] Saved GeoJSON → {geojson_path}")

    # Save Shapefile (WGS84) — untuk kompatibilitas script 04_finetune_indonesia.py
    shp_path = OUT_DIR / "dki_kumuh.shp"
    gdf_wgs.to_file(shp_path, driver="ESRI Shapefile")
    print(f"        Saved Shapefile → {shp_path}")

    # Summary
    print("\n=== SUMMARY ===")
    print(gdf_wgs.groupby("KAB_NAME").agg(
        polygon_count=("FID", "count"),
        total_luas_ha=("Luas_Ha", "sum"),
    ))
    print(f"\nTotal polygon: {len(gdf_wgs)}")
    print(f"Total luas: {gdf_wgs['Luas_Ha'].sum():.2f} Ha")
    print(f"\nBounding box (WGS84): {gdf_wgs.total_bounds}")
    print("  → pakai bbox ini untuk filter Sentinel-2 di GEE")


if __name__ == "__main__":
    main()