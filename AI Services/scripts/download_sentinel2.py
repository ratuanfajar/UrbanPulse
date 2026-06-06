"""
Download Sentinel-2 per wilayah sesuai data KOTAKU DKI Jakarta via GEE.
Output: dataset/Indonesia/imagery/per_wilayah/<wilayah>.tif
"""
import ee
import requests
import os
import warnings
import sys
from pathlib import Path

# Tambahkan root project ke path agar bisa import list_wilayah
sys.path.append(str(Path(__file__).parent.parent))
from list_wilayah import DAFTAR_WILAYAH

warnings.filterwarnings("ignore")

# === Konfigurasi ===
PROJECT_ID    = "exemplary-oven-352202"
OUTPUT_FOLDER = Path(__file__).parent.parent / "dataset" / "Indonesia" / "imagery" / "per_wilayah"
DATE_START    = "2018-05-01"
DATE_END      = "2018-09-30"
CLOUD_MAX     = 20
BANDS         = ["B2", "B3", "B4", "B8A", "B11", "B12"]

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# === Inisialisasi GEE ===
try:
    ee.Initialize(project=PROJECT_ID)
    print(f"✅ Terhubung ke GEE Project: {PROJECT_ID}")
except Exception:
    print("🔐 Perlu autentikasi dulu...")
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)
    print(f"✅ Terhubung ke GEE Project: {PROJECT_ID}")

def mask_clouds(image):
    scl = image.select("SCL")
    bad = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
    return image.updateMask(bad.Not())

print(f"\n🚀 Memulai download untuk {len(DAFTAR_WILAYAH)} wilayah...")
print("-" * 50)

for nama, coords in DAFTAR_WILAYAH.items():
    out_path = OUTPUT_FOLDER / f"{nama}.tif"

    if out_path.exists():
        print(f"⏭  Skip {nama.upper()} (file sudah ada: {out_path.stat().st_size/1e6:.1f} MB)")
        continue

    print(f"🛰️  Memproses: {nama.upper()}...")

    try:
        aoi = ee.Geometry.Rectangle(coords)

        image = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(DATE_START, DATE_END)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_MAX))
            .map(mask_clouds)
            .select(BANDS)
            .median()
            .clip(aoi)
        )

        # Normalize ke 0-1 (float32) — lebih mudah untuk model AI
        final_img = image.divide(10000)

        download_url = final_img.getDownloadURL({
            "name": nama,
            "scale": 10,
            "crs": "EPSG:4326",
            "format": "GEO_TIFF",
            "region": aoi.toGeoJSONString(),
            "bands": BANDS,
        })

        print(f"   📥 Downloading...")
        r = requests.get(download_url, stream=True, timeout=300)
        r.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

        print(f"   ✅ Selesai: {nama}.tif ({out_path.stat().st_size/1e6:.1f} MB)")

    except Exception as e:
        print(f"   ❌ Error pada {nama}: {e}")

print("-" * 50)
print("✨ SEMUA TUGAS SELESAI!")
print(f"📂 Output folder: {OUTPUT_FOLDER.resolve()}")