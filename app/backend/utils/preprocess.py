"""
utils/preprocess.py
Fungsi preprocessing gambar sebelum diprediksi oleh model.
"""

import numpy as np
from PIL import Image
import io


IMG_SIZE = (224, 224)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Menerima bytes gambar, lakukan preprocessing:
    - Buka gambar dengan PIL
    - Convert ke RGB (handle PNG/RGBA)
    - Resize ke 224x224
    - Normalisasi pixel ke [0, 1]
    - Tambah batch dimension

    Returns:
        np.ndarray shape (1, 224, 224, 3)
    """
    # Buka gambar dari bytes
    img = Image.open(io.BytesIO(image_bytes))

    # Pastikan format RGB (handle grayscale, RGBA, dll)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize ke ukuran yang dibutuhkan model
    img = img.resize(IMG_SIZE, Image.LANCZOS)

    # Convert ke numpy array
    img_array = np.array(img, dtype=np.float32)

    # Normalisasi: pixel [0, 255] → [0.0, 1.0]
    img_array = img_array / 255.0

    # Tambah batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
