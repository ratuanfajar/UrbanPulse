# list_kecamatan.py
# Auto-generated dari dki_kumuh.geojson
# Format: [min_lon, min_lat, max_lon, max_lat]

import geopandas as gpd
from pathlib import Path

def get_daftar_wilayah():
    geojson_path = Path(__file__).parent.parent / "dataset" / "Indonesia" / "labels" / "dki_kumuh.geojson"
    gdf = gpd.read_file(geojson_path).to_crs("EPSG:4326")
    
    result = {}
    for kab_name, group in gdf.groupby("KAB_NAME"):
        safe_name = kab_name.lower().replace(" ", "_")
        bounds = group.total_bounds  # [minx, miny, maxx, maxy]
        result[safe_name] = bounds.tolist()
    
    return result

DAFTAR_WILAYAH = get_daftar_wilayah()

if __name__ == "__main__":
    for nama, coords in DAFTAR_WILAYAH.items():
        print(f"{nama}: {coords}")