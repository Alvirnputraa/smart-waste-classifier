# 🗑️ Smart Waste Classifier

Sistem klasifikasi sampah berbasis web menggunakan **Machine Learning** (MobileNetV2 Transfer Learning) untuk membedakan sampah **Organik** dan **Anorganik**.

> **Mata Kuliah:** Machine Learning — Informatika

---

## 👥 Anggota Kelompok 3

| No | Nama | NIM |
|----|------|-----|
| 1 | Alvi Al Virana Putra | 2310614033 |
| 2 | D'jihni Basil Fauzani | 2310614039 |
| 3 | Rival Syifa Ruslani | 2310614036 |

---

## 📋 Deskripsi Proyek

Sistem ini mengklasifikasikan gambar sampah menjadi dua kategori:
- 🌿 **Organik** — sisa makanan, daun, ranting, bahan alami
- ♻️ **Anorganik** — plastik, logam, kaca, kertas kering

### Fitur Utama
- Upload gambar dari file
- Ambil gambar langsung dari kamera HP
- Prediksi real-time dengan confidence score
- UI modern responsive (dark/light)
- Deployed via Cloudflare Tunnel (HTTPS)

---

## 🛠️ Tech Stack

| Layer | Teknologi |
|-------|-----------|
| Frontend | React 18 + Vite (JavaScript) |
| Backend | FastAPI + Uvicorn (Python) |
| ML Model | TensorFlow / Keras |
| Architecture | MobileNetV2 Transfer Learning |
| Image Processing | Pillow (PIL) |
| Web Server | Nginx |
| Deployment | Ubuntu Server + Cloudflare Tunnel |

---

## 📁 Struktur Folder

```
smart-waste-classifier/
├── data/               # Dataset (link ke Google Drive/Kaggle)
├── notebooks/          # Jupyter notebook (EDA, training)
├── models/             # Saved model (.h5) + class indices
├── app/                # Deployment app (FastAPI + React)
│   ├── backend/        # FastAPI server + training script
│   └── frontend/       # React + Vite UI
├── reports/            # Laporan PDF dan slide PDF
├── requirements.txt    # Dependensi Python
└── README.md
```

---

## 📊 Hasil Performa Model

| Metrik | Nilai |
|--------|-------|
| **Validation Accuracy** | 94.41% |
| **Validation Loss** | 0.2218 |
| **Arsitektur** | MobileNetV2 + Custom Head |
| **Optimizer** | Adam (lr=1e-3 → 1e-4) |
| **Loss Function** | Binary Crossentropy |
| **Training Strategy** | 2-Phase (Frozen → Fine-tune) |

### Dataset
- **Organic (O):** 12.565 gambar
- **Anorganik (R):** 9.999 gambar
- **Total:** 22.564 gambar
- **Split:** 80% training, 20% validation

### Preprocessing
- Resize: 224×224 piksel
- Normalisasi: pixel / 255.0
- Augmentasi: rotation, zoom, flip, shift, brightness

---

## ⚙️ Cara Instalasi & Run

### Prasyarat
- Python 3.8 – 3.10
- Node.js 18+
- pip

### 1. Clone Repository
```bash
git clone https://github.com/Alvirnputraa/smart-waste-classifier.git
cd smart-waste-classifier
```

### 2. Setup Backend
```bash
cd app/backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

### 3. Training Model
```bash
python train_model.py
```
> Output: `models/waste_classifier.h5` (accuracy ~94%)

### 4. Jalankan Backend
```bash
python main.py
```
> Backend berjalan di: http://localhost:8000

### 5. Setup & Jalankan Frontend
```bash
cd app/frontend
npm install
npm run dev
```
> Frontend berjalan di: http://localhost:5173

### 6. Buka Browser
```
http://localhost:5173
```

---

## 🚀 Demo Online

**URL:** https://smart-waste.shaleh.live

---

## 📓 Training di Google Colab

Notebook training tersedia di folder `notebooks/`:
1. Upload dataset ke Google Drive (`MyDrive/DATASET/TRAIN/`)
2. Buka notebook di Google Colab
3. Pilih Runtime → GPU (T4)
4. Jalankan semua cell

---

## 📦 Dataset

**Sumber:** [Kaggle - Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)

**Lisensi:** CC0 Public Domain — bebas digunakan untuk keperluan akademik.

---

## 🔌 API Endpoint

### `POST /predict`
Upload gambar untuk diprediksi.

**Request:** `multipart/form-data` dengan field `file`

**Response:**
```json
{
  "prediction": "Organik",
  "confidence": 95.42
}
```

---

## 📄 Lisensi

Project ini dibuat untuk keperluan tugas akademik Mata Kuliah Machine Learning — Program Studi Informatika.
