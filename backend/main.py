"""
main.py
FastAPI backend untuk Smart Waste Classifier.
Endpoint: POST /predict — menerima gambar, return prediksi Organic/Recyclable.
"""

import os
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # pakai Keras 2 untuk baca .h5
tf.get_logger().setLevel('ERROR')

from utils.preprocess import preprocess_image

# ─────────────────────────────────────────────
# INISIALISASI APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Smart Waste Classifier API",
    description="API untuk klasifikasi sampah: Organic vs Recyclable",
    version="1.0.0",
)

# CORS — izinkan semua origin (Cloudflare Tunnel pakai domain dinamis)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "waste_classifier.h5")
CLASS_INDICES_PATH = os.path.join(os.path.dirname(__file__), "model", "class_indices.json")

model = None
class_labels = {}  # {0: "Organic", 1: "Recyclable"} atau sesuai training


def load_model():
    """Load model saat startup."""
    global model, class_labels

    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Model tidak ditemukan di: {MODEL_PATH}")
        print("   Jalankan train_model.py terlebih dahulu.")
        return

    print(f"✅ Loading model dari: {MODEL_PATH}")
    try:
        # Coba load dengan tf_keras (kompatibel .h5 dari TF 2.x lama)
        import tf_keras
        model = tf_keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model dimuat via tf_keras")
    except Exception:
        # Fallback ke tf.keras biasa
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model dimuat via tf.keras")

    print(f"✅ TensorFlow version: {tf.__version__}")
    print("✅ Model berhasil dimuat.")

    # Load class indices jika ada
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, "r") as f:
            raw = json.load(f)
        # raw = {"O": 0, "R": 1} → balik jadi {0: "Organic", 1: "Recyclable"}
        label_map = {"O": "Organik", "R": "Anorganik"}
        class_labels = {v: label_map.get(k, k) for k, v in raw.items()}
        print(f"✅ Class labels: {class_labels}")
    else:
        # Default fallback
        class_labels = {0: "Organik", 1: "Anorganik"}


# Load model saat startup
load_model()


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "Smart Waste Classifier API is running",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    """Cek status model."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "class_labels": class_labels,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint prediksi sampah.

    - Menerima: file gambar (jpg/png/webp)
    - Return: prediksi label + confidence score
    """
    # Validasi model sudah dimuat
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model belum dimuat. Jalankan train_model.py terlebih dahulu.",
        )

    # Validasi tipe file
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipe file tidak didukung: {file.content_type}. Gunakan JPG/PNG/WEBP.",
        )

    try:
        # Baca bytes gambar
        image_bytes = await file.read()

        # Preprocessing
        img_array = preprocess_image(image_bytes)

        # Prediksi
        prediction = model.predict(img_array, verbose=0)
        score = float(prediction[0][0])  # nilai sigmoid [0.0 - 1.0]

        # Interpretasi hasil
        # score mendekati 0 → kelas index 0, score mendekati 1 → kelas index 1
        # Sesuaikan dengan class_indices dari training
        # Jika O=0, R=1: score < 0.5 → Organic, score >= 0.5 → Recyclable
        if score >= 0.5:
            predicted_index = 1
            confidence = score * 100
        else:
            predicted_index = 0
            confidence = (1 - score) * 100

        predicted_label = class_labels.get(predicted_index, "Unknown")

        return JSONResponse(
            content={
                "prediction": predicted_label,
                "confidence": round(confidence, 2),
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat prediksi: {str(e)}",
        )


# ─────────────────────────────────────────────
# JALANKAN SERVER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
