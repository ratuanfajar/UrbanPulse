"""Scraping delineasi kumuh & batas administrasi KOTAKU dari webmap ArcGIS."""
from pathlib import Path
import time

import requests
import geopandas as gpd
from arcgis2geojson import arcgis2geojson

DATA_URL = "https://www.arcgis.com/sharing/rest/content/items/{item_id}/data?f=json"
SRC_CRS, DST_CRS = "EPSG:3857", "EPSG:4326"
OUT_DIR = Path("data/raw")

CITIES = {
    "ambon": {
        "item_id": "5867a6108c49455897abda87d719b8d1",
        "layers": {"Delineasi Kumuh Akhir": "positif",
                   "Wilayah Administrasi": "admin",
                   "Profil Kumuh Basis": "profil"},
    },
    "dki": {
        "item_id": "de25f7e11e3444c694027dafd62cfad3",
        "layers": {"Sebaran RW Kumuh": "positif",
                   "Kelurahan": "admin",
                   "Kecamatan": "admin_kec"},
    },
    "samarinda": {
        "item_id": "eb55085c5c7a42f2bc33519fc14c1267",
        "layers": {"Delineasi Kumuh 2020": "positif",
                   "Batas Administrasi AR": "admin"},
    },
    "kebumen": {
        "item_id": "99c191e2b88544d6b2fe660442bdb657",
        "layers": {"Status Kumuh Akhir 2020": "positif",
                   "Wilayah Administrasi": "admin",
                   "Profil Kumuh Basis": "profil"},
    },
}


def fetch_webmap(item_id: str, retries: int = 3) -> dict:
    headers = {"User-Agent": "Mozilla/5.0 (kotaku-scraper)"}
    for attempt in range(retries):
        try:
            r = requests.get(DATA_URL.format(item_id=item_id), headers=headers, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def find_featureset(webmap: dict, title: str) -> dict | None:
    for layer in webmap.get("operationalLayers", []):
        if layer.get("title") == title:
            sub = (layer.get("featureCollection", {}).get("layers") or [None])[0]
            return sub.get("featureSet") if sub else None
    return None


def to_gdf(featureset: dict) -> gpd.GeoDataFrame | None:
    features = [f for f in featureset.get("features", []) if f.get("geometry")]
    if not features:
        return None
    geojson = arcgis2geojson({"features": features})
    gdf = gpd.GeoDataFrame.from_features(geojson["features"])
    return gdf.set_crs(SRC_CRS).to_crs(DST_CRS)


def scrape_city(city: str, config: dict) -> list[tuple[str, str, int]]:
    webmap = fetch_webmap(config["item_id"])
    city_dir = OUT_DIR / city
    city_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for title, role in config["layers"].items():
        featureset = find_featureset(webmap, title)
        gdf = to_gdf(featureset) if featureset else None
        if gdf is None or gdf.empty:
            print(f"  [!] {city}/{title}: tidak ditemukan / kosong")
            continue
        gdf["city"], gdf["source_layer"], gdf["role"] = city, title, role
        slug = title.lower().replace(" ", "_").replace("/", "_")
        gdf.to_file(city_dir / f"{role}__{slug}.geojson", driver="GeoJSON")
        results.append((role, title, len(gdf)))
    return results


def main() -> None:
    summary = []
    for city, config in CITIES.items():
        for role, title, n in scrape_city(city, config):
            summary.append((city, role, title, n))
            print(f"  {city:10s} | {role:10s} | {title:28s} | {n}")
    raw_positif = sum(n for _, role, _, n in summary if role == "positif")
    print(f"\n  Poligon delineasi: {raw_positif}")

if __name__ == "__main__":
    main()