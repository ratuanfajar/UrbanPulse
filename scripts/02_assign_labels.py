"""Bangun tabel label biner (slum/non-slum) per kelurahan dari delineasi kumuh KOTAKU."""
from pathlib import Path

import geopandas as gpd
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")

SNAPSHOT = "awal"               # "awal" atau "akhir"
SLUM_AREA_FRAC_THRESHOLD = 0.005
SEVERITY = {"ringan": 1, "sedang": 2, "berat": 3}
SEVERITY_LABEL = {1: "Ringan", 2: "Sedang", 3: "Berat"}
NAME_FIELDS = ["Kelurahan", "KEL_NAME", "NAMOBJ", "NAMA", "Desa_Kel",
               "Kel", "name", "Nama_Kel", "WADMKK"]

CITIES = {
    "ambon": {
        "admin": "admin__wilayah_administrasi.geojson",
        "kumuh": "positif__delineasi_kumuh_akhir.geojson",
        "status": {"awal": "kumuh_awal", "akhir": "kumuh_akhi"},
    },
    "dki": {
        "admin": "admin__kelurahan.geojson",
        "kumuh": "positif__sebaran_rw_kumuh.geojson",
        "status": {"awal": "STATUS", "akhir": "STATUS"},
    },
    "samarinda": {
        "admin": "admin__batas_administrasi_ar.geojson",
        "kumuh": "positif__delineasi_kumuh_2020.geojson",
        "status": {"awal": "kum_awal", "akhir": "kum_akhir"},
    },
    "kebumen": {
        "admin": "admin__wilayah_administrasi.geojson",
        "kumuh": "positif__status_kumuh_akhir_2020.geojson",
        "status": {"awal": "kum_awal", "akhir": "kum_akhir"},
    },
}


def is_kumuh(value) -> bool:
    text = str(value).lower()
    return "kumuh" in text and "tidak" not in text


def severity_rank(value) -> int:
    text = str(value).lower()
    return next((rank for key, rank in SEVERITY.items() if key in text), 1)


def union_geometry(gdf: gpd.GeoDataFrame):
    try:
        return gdf.geometry.union_all()
    except AttributeError:
        return gdf.geometry.unary_union


def unit_name(row) -> str | None:
    for field in NAME_FIELDS:
        if field in row and pd.notna(row[field]) and str(row[field]).strip():
            return str(row[field]).strip()
    return None


def label_city(city: str, config: dict) -> gpd.GeoDataFrame:
    admin = gpd.read_file(RAW / city / config["admin"]).to_crs("EPSG:4326")
    kumuh = gpd.read_file(RAW / city / config["kumuh"]).to_crs("EPSG:4326")

    status_field = config["status"][SNAPSHOT]
    kumuh = kumuh[kumuh[status_field].apply(is_kumuh)].copy()
    kumuh["severity"] = kumuh[status_field].apply(severity_rank)

    crs_m = admin.estimate_utm_crs()
    admin_m = admin.to_crs(crs_m).copy()
    kumuh_m = kumuh.to_crs(crs_m).copy()
    admin_m["geometry"] = admin_m.geometry.buffer(0)
    kumuh_m["geometry"] = kumuh_m.geometry.buffer(0)

    admin_m["unit_id"] = [f"{city}_{i:04d}" for i in range(len(admin_m))]
    admin_m["unit_area_m2"] = admin_m.geometry.area

    if len(kumuh_m):
        kumuh_union = union_geometry(kumuh_m)
        admin_m["kumuh_area"] = admin_m.geometry.intersection(kumuh_union).area
        joined = gpd.sjoin(admin_m[["unit_id", "geometry"]],
                           kumuh_m[["severity", "geometry"]],
                           predicate="intersects", how="left")
        severity = joined.groupby("unit_id")["severity"].max()
        admin_m = admin_m.merge(severity.rename("severity"), on="unit_id", how="left")
    else:
        admin_m["kumuh_area"] = 0.0
        admin_m["severity"] = pd.NA

    admin_m["slum_area_frac"] = (admin_m["kumuh_area"] / admin_m["unit_area_m2"]).clip(0, 1)
    admin_m["slum"] = (admin_m["slum_area_frac"] > SLUM_AREA_FRAC_THRESHOLD).astype(int)
    admin_m["kumuh_level"] = admin_m["severity"].map(SEVERITY_LABEL).where(
        admin_m["slum"] == 1, "Non-Slum")

    return gpd.GeoDataFrame({
        "unit_id": admin_m["unit_id"].values,
        "unit_name": admin.apply(unit_name, axis=1).values,
        "city": city,
        "unit_area_m2": admin_m["unit_area_m2"].values,
        "slum_area_frac": admin_m["slum_area_frac"].values,
        "slum": admin_m["slum"].values,
        "kumuh_level": admin_m["kumuh_level"].values,
        "geometry": admin.geometry.values,
    }, crs="EPSG:4326")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = [label_city(city, config) for city, config in CITIES.items()]

    for part in parts:
        city = part["city"].iloc[0]
        total, slum = len(part), int(part["slum"].sum())
        print(f"  {city:10s} | unit={total:4d} | slum={slum:4d} | "
              f"non-slum={total - slum:4d} | rasio={slum / total:.1%}")

    labels = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    labels.to_file(OUT / "labels.geojson", driver="GeoJSON")

    total, slum = len(labels), int(labels["slum"].sum())
    print(f"\n  TOTAL    | unit={total} | slum={slum} | "
          f"non-slum={total - slum} | rasio={slum / total:.1%} (SNAPSHOT='{SNAPSHOT}')")


if __name__ == "__main__":
    main()