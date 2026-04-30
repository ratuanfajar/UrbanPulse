# UrbanPulse — Sistem Deteksi Lingkungan Kumuh Berbasis Citra Satelit

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-CUDA%2012.1-red?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/Next.js-16.2-black?style=flat-square&logo=next.js" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square" />
</p>

> Proyek ini dikembangkan dalam rangka kompetisi Datathon. UrbanPulse adalah sistem end-to-end untuk mendeteksi dan memvisualisasikan kawasan kumuh di perkotaan Indonesia menggunakan citra satelit Sentinel-2 dan model deep learning berbasis transfer learning.

---

## Latar Belakang

Kawasan kumuh (*slum area*) merupakan salah satu tantangan urban yang signifikan di Indonesia, terutama di kota-kota besar seperti Jakarta. Identifikasi kawasan kumuh secara manual membutuhkan waktu dan sumber daya yang besar. Proyek ini memanfaatkan citra satelit Sentinel-2 dan model segmentasi semantik berbasis deep learning untuk mengotomatisasi proses deteksi tersebut secara skalabel dan akurat.

---

## Struktur Proyek

```
UrbanPulse/
├── AI-Services/                 # Pipeline Machine Learning
│   ├── notebook/                # Jupyter notebooks (EDA, training, evaluasi)
│   ├── scripts/                 # Script download dan preprocessing data
│   ├── processed/               # Dataset yang telah diproses
│   ├── outputs_phase1/          # Output dan metrik Phase 1
│   ├── outputs_phase2/          # Output dan metrik Phase 2
│   ├── training_outputs/        # Checkpoint model terlatih
│   ├── pyproject.toml           # Dependensi Python
│   └── .python-version          # Versi Python yang digunakan
│
└── Web-Apps/                    # Aplikasi Frontend
    ├── app/                     # Next.js app directory
    ├── components/              # Komponen React
    ├── public/                  # Aset statis
    ├── utils/                   # Fungsi utilitas
    ├── package.json             # Dependensi JavaScript
    ├── .env.example             # Template environment variables
    └── README.md                # Dokumentasi Web Apps
```

---

## Pendekatan & Metodologi

### Model
Proyek ini menggunakan **Prithvi-EO-2.0-100M-TL** — foundation model geospasial dari NASA/IBM yang telah dilatih pada data *Harmonized Landsat Sentinel-2 (HLS)*. Model ini di-*fine-tune* menggunakan arsitektur encoder-decoder via **TerraTorch** untuk tugas segmentasi semantik biner (kumuh vs. non-kumuh).

### Dataset
| Dataset | Sumber | Jumlah | Keterangan |
|--------|--------|--------|------------|
| Argentina Slum | Kaggle | ~45.000 patch TIF | Resolusi 10m, 4 band, patch 32×32 |
| Mumbai Slum | Kaggle | 1 scene besar | Sentinel-2, label `.npy` biner |
| Indonesia (Kotaku) | ArcGIS / Data DKI | 139 TIF + GeoJSON | Per-RW, berlabel, domain target |

### Strategi Transfer Learning
1. **Pre-training domain** — Argentina & Mumbai digunakan untuk mengajarkan pola visual umum kawasan kumuh
2. **Fine-tuning target domain** — Data Indonesia (Kotaku) sebagai domain target utama untuk konteks perkotaan Indonesia

### Metrik Evaluasi
- **IoU per kelas** (Intersection over Union)
- **mIoU** (mean IoU)
- **F1-Score kelas kumuh**
- **Pixel Accuracy**
- **Precision & Recall kelas kumuh**

---

## Tech Stack

### AI Services
| Kategori | Library |
|---------|---------|
| Deep Learning | PyTorch, TerraTorch |
| Computer Vision | Albumentations, segmentation-models-pytorch |
| Geospasial | Rasterio, GeoPandas, Shapely |
| Data Processing | NumPy, Pandas, Xarray, Dask |
| Citra Satelit | Sentinel-2 via Planetary Computer |
| Visualisasi | Matplotlib, Seaborn, Plotly |

### Web Apps
| Kategori | Teknologi |
|---------|-----------|
| Framework | Next.js 16.2 |
| UI | React 19, Tailwind CSS 4 |
| Peta | Azure Maps Control |
| Icons | Lucide React |

---

## Cara Menjalankan

### Prasyarat

**AI Services:**
- Python 3.11–3.12
- NVIDIA GPU dengan CUDA 12.1 (sangat direkomendasikan)
- UV package manager

**Web Apps:**
- Node.js 18+
- npm

---

### AI Services

```bash
cd AI-Services

# Install dependensi dengan UV
uv sync

# Download data Kotaku (kawasan kumuh Indonesia)
python scripts/download_kotaku.py

# Download citra satelit Sentinel-2
python scripts/download_sentinel2.py

# Jalankan notebook analisis
jupyter notebook notebook/
```

---

### Web Apps

```bash
cd Web-Apps

# Install dependensi
npm install

# Salin dan isi environment variables
cp .env.example .env.local
# Edit .env.local → isi NEXT_PUBLIC_AZURE_MAPS_KEY

# Jalankan server development
npm run dev
```

Aplikasi akan berjalan di `http://localhost:3000`

---

## 🔧 Konfigurasi Environment

Buat file `.env.local` di dalam folder `Web-Apps/`:

```env
NEXT_PUBLIC_AZURE_MAPS_KEY=your_azure_maps_api_key
NEXT_PUBLIC_API_BASE_URL=your_api_endpoint_url
```

---

## Fase Pengembangan

### Phase 1 — Eksplorasi & Baseline
- EDA ketiga dataset (Argentina, Mumbai, Indonesia)
- Analisis kompatibilitas band dan resolusi
- Pelatihan model baseline
- Output tersimpan di `outputs_phase1/`

### Phase 2 — Fine-Tuning & Optimasi
- Fine-tuning Prithvi-EO-2.0 dengan data Indonesia sebagai target domain
- Augmentasi data dan penanganan class imbalance
- Evaluasi metrik IoU, F1, mIoU
- Output tersimpan di `outputs_phase2/`

---

## Fitur Aplikasi Web

- Visualisasi peta interaktif hasil prediksi kawasan kumuh menggunakan Azure Maps
- Filter berdasarkan wilayah administratif (kecamatan, kelurahan, RW)
- Tampilan responsif berbasis Tailwind CSS
- Eksplorasi data geografis secara real-time

---

## Troubleshooting

**PyTorch gagal load:**
```bash
# Sesuaikan dengan versi CUDA di nvidia-smi
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Web App build error:**
```bash
rm -rf node_modules && npm install
rm -rf .next
```

**Azure Maps tidak muncul:**
- Pastikan `NEXT_PUBLIC_AZURE_MAPS_KEY` sudah diisi dengan benar di `.env.local`
- Cek console browser untuk error CORS

---

## Referensi

- [Prithvi-EO-2.0 (NASA/IBM)](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL)
- [TerraTorch Documentation](https://terratorch.readthedocs.io)
- [Planetary Computer](https://planetarycomputer.microsoft.com)
- [Data Kotaku — Kementerian PUPR](https://kotaku.pu.go.id)
- [Azure Maps Documentation](https://learn.microsoft.com/en-us/azure/azure-maps)
- [Next.js Documentation](https://nextjs.org/docs)

---

## Lisensi

Proyek ini dibuat untuk keperluan kompetisi Datathon. Silakan merujuk pada ketentuan kompetisi untuk informasi lisensi lebih lanjut.