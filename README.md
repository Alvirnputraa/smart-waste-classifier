# 🗑️ Smart Waste Classifier

Aplikasi klasifikasi sampah berbasis web menggunakan **Machine Learning** (MobileNetV2 Transfer Learning).

> Tugas Mata Kuliah Machine Learning — Informatika

---

## 📋 Deskripsi

Sistem ini mengklasifikasikan gambar sampah menjadi dua kategori:
- 🌿 **Organic** — sisa makanan, daun, ranting, bahan alami
- ♻️ **Recyclable** — plastik, logam, kaca, kertas kering

---

## 🛠️ Tech Stack

| Layer | Teknologi |
|-------|-----------|
| Frontend | React 18 + Vite |
| Backend | FastAPI (Python) |
| ML Model | TensorFlow / Keras |
| Architecture | MobileNetV2 Transfer Learning |
| Image Processing | PIL (Pillow) |
| Styling | CSS Modules (dark modern UI) |

---

## 📁 Struktur Project

```
smart-waste/
│
├── backend/
│   ├── main.py              ← FastAPI server + endpoint /predict
│   ├── train_model.py       ← Script training model
│   ├── requirements.txt     ← Dependensi Python
│   ├── model/
│   │   ├── waste_classifier.h5     ← Model hasil training (dibuat setelah train)
│   │   └── class_indices.json      ← Mapping kelas
│   └── utils/
│       ├── __init__.py
│       └── preprocess.py    ← Fungsi preprocessing gambar
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx          ← Komponen utama React
│   │   ├── App.module.css   ← Styling CSS Modules
│   │   ├── main.jsx         ← Entry point React
│   │   └── index.css        ← Global CSS variables
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
└── README.md
```

---

## ⚙️ Cara Install & Menjalankan

### Prasyarat
- Python 3.9 – 3.11
- Node.js 18+
- pip

---

### 1️⃣ Setup Backend

Buka terminal, masuk ke folder backend:

```bash
cd smart-waste/backend
```

Buat virtual environment (disarankan):

```bash
python -m venv venv
```

Aktifkan virtual environment:

```bash
# Windows CMD
venv\Scripts\activate

# Windows PowerShell
venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate
```

Install dependensi:

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Training Model

> ⚠️ Lakukan ini SEBELUM menjalankan backend server.
> Proses training membutuhkan waktu 10–30 menit tergantung spesifikasi PC.

```bash
python train_model.py
```

Output yang diharapkan:
```
[1/5] Loading dataset...
  Class mapping: {'O': 0, 'R': 1}
  Training samples  : 18051
  Validation samples: 4513

[2/5] Building MobileNetV2 model...
[3/5] Training model (Phase 1 - Frozen base)...
[4/5] Fine-tuning (Phase 2 - Unfreeze top 30 layers)...
[5/5] Evaluating model...

  ✅ Final Validation Accuracy : 95.xx%
  ✅ Final Validation Loss     : 0.xxxx

  Model saved to: model/waste_classifier.h5
```

File yang dihasilkan:
- `model/waste_classifier.h5` — model terlatih
- `model/class_indices.json` — mapping kelas
- `model/training_history.png` — grafik akurasi & loss

---

### 3️⃣ Jalankan Backend Server

```bash
python main.py
```

Atau dengan uvicorn langsung:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend berjalan di: **http://localhost:8000**

Cek API docs: **http://localhost:8000/docs**

---

### 4️⃣ Setup & Jalankan Frontend

Buka terminal baru, masuk ke folder frontend:

```bash
cd smart-waste/frontend
```

Install dependensi Node.js:

```bash
npm install
```

Jalankan development server:

```bash
npm run dev
```

Frontend berjalan di: **http://localhost:5173**

---

## 🚀 Cara Penggunaan

1. Buka browser → **http://localhost:5173**
2. Pilih tab **Upload Gambar** atau **Kamera**
3. Upload/ambil foto sampah
4. Klik tombol **🔍 Prediksi Sampah**
5. Lihat hasil:
   - **Prediction**: Organic / Recyclable
   - **Confidence**: persentase keyakinan model

---

## 📊 Contoh Output

```json
{
  "prediction": "Organic",
  "confidence": 95.42
}
```

```json
{
  "prediction": "Recyclable",
  "confidence": 88.17
}
```

---

## 🔌 API Endpoint

### `POST /predict`

Upload gambar untuk diprediksi.

**Request:**
```
Content-Type: multipart/form-data
Body: file = <image file>
```

**Response:**
```json
{
  "prediction": "Organic",
  "confidence": 95.42
}
```

### `GET /health`

Cek status server dan model.

### `GET /`

Health check dasar.

---

## ⚠️ Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `Model belum dimuat` | Jalankan `train_model.py` terlebih dahulu |
| `Cannot connect to server` | Pastikan backend berjalan di port 8000 |
| `CUDA/GPU error` | Training akan otomatis fallback ke CPU |
| `npm install` gagal | Pastikan Node.js versi 18+ terinstall |
| Kamera tidak muncul | Izinkan akses kamera di browser |

---

## 📦 Dataset

Dataset berada di:
```
C:\Users\Administrator\Downloads\archive (1)\DATASET\TRAIN\
├── O\   ← 12.565 gambar Organic
└── R\   ← 9.999 gambar Recyclable
```

Total: **~22.564 gambar**

---

## 👨‍💻 Dibuat untuk

Tugas Mata Kuliah **Machine Learning** — Program Studi Informatika
