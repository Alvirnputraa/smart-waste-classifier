# ============================================================
# SMART WASTE CLASSIFIER — Google Colab Notebook
# Tugas Machine Learning — Informatika
# Model: MobileNetV2 Transfer Learning
# Klasifikasi: Organik vs Anorganik
# ============================================================

# ============================================================
# CELL 1: Upload Dataset ke Google Drive
# ============================================================
# Upload folder dataset ke Google Drive kamu dengan struktur:
# MyDrive/DATASET/TRAIN/
#   ├── O/   (gambar organik)
#   └── R/   (gambar anorganik)
#
# Lalu mount Google Drive:

from google.colab import drive
drive.mount('/content/drive')


# ============================================================
# CELL 2: Import Library
# ============================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print("TensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)


# ============================================================
# CELL 3: Konfigurasi
# ============================================================

DATASET_DIR = "/content/drive/MyDrive/DATASET/TRAIN"
MODEL_SAVE_PATH = "/content/model/waste_classifier.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_P1 = 15    # Phase 1: frozen base
EPOCHS_P2 = 10    # Phase 2: fine-tune
VAL_SPLIT = 0.2
LR_P1 = 1e-3
LR_P2 = 1e-4

os.makedirs("/content/model", exist_ok=True)

# Validasi dataset
valid_classes = ['O', 'R']
for cls in valid_classes:
    cls_path = os.path.join(DATASET_DIR, cls)
    n = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    label = "Organik" if cls == "O" else "Anorganik"
    print(f"  {cls} ({label}): {n} gambar")


# ============================================================
# CELL 4: Data Generator (Preprocessing + Augmentasi)
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
    validation_split=VAL_SPLIT,
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=VAL_SPLIT,
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=valid_classes,
    subset="training",
    shuffle=True,
    seed=42,
)

val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=valid_classes,
    subset="validation",
    shuffle=False,
    seed=42,
)

print(f"\nClass mapping : {train_gen.class_indices}")
print(f"Train samples : {train_gen.samples}")
print(f"Val samples   : {val_gen.samples}")


# ============================================================
# CELL 5: Visualisasi Sample Data
# ============================================================

images, labels = next(train_gen)
plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i])
    label = "Organik" if labels[i] == 0 else "Anorganik"
    plt.title(label, fontsize=10)
    plt.axis('off')
plt.suptitle("Sample Data Training (Setelah Augmentasi)", fontsize=13)
plt.tight_layout()
plt.show()


# ============================================================
# CELL 6: Build Model MobileNetV2
# ============================================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_P1),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ============================================================
# CELL 7: Training Phase 1 (Frozen Base)
# ============================================================

callbacks_p1 = [
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
]

steps_per_epoch = train_gen.samples // BATCH_SIZE
validation_steps = val_gen.samples // BATCH_SIZE

print(f"Phase 1: Training head (base frozen, {EPOCHS_P1} epochs)")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

history1 = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS_P1,
    callbacks=callbacks_p1,
    verbose=1,
)


# ============================================================
# CELL 8: Training Phase 2 (Fine-Tuning)
# ============================================================

base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_P2),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks_p2 = [
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-8,
        verbose=1,
    ),
]

print(f"\nPhase 2: Fine-tuning (unfreeze top 40 layers, {EPOCHS_P2} epochs)")

history2 = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS_P2,
    callbacks=callbacks_p2,
    verbose=1,
)


# ============================================================
# CELL 9: Evaluasi Model
# ============================================================

val_gen.reset()
val_loss, val_acc = model.evaluate(val_gen, steps=validation_steps, verbose=0)

print(f"\n{'='*50}")
print(f"  HASIL EVALUASI MODEL")
print(f"{'='*50}")
print(f"  Validation Accuracy : {val_acc * 100:.2f}%")
print(f"  Validation Loss     : {val_loss:.4f}")
print(f"{'='*50}")
print(f"\n  Model disimpan di: {MODEL_SAVE_PATH}")


# ============================================================
# CELL 10: Plot Training History
# ============================================================

acc       = history1.history["accuracy"]      + history2.history["accuracy"]
val_acc_  = history1.history["val_accuracy"]  + history2.history["val_accuracy"]
loss_     = history1.history["loss"]          + history2.history["loss"]
val_loss_ = history1.history["val_loss"]      + history2.history["val_loss"]

epochs_range = range(1, len(acc) + 1)
phase1_end   = len(history1.history["accuracy"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Smart Waste Classifier — Training History", fontsize=14, fontweight='bold')

# Accuracy
axes[0].plot(epochs_range, acc,      label="Train Accuracy", color="#4ade80")
axes[0].plot(epochs_range, val_acc_, label="Val Accuracy",   color="#22d3ee")
axes[0].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.5, label="Fine-tune start")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0.5, 1.0])

# Loss
axes[1].plot(epochs_range, loss_,     label="Train Loss", color="#f97316")
axes[1].plot(epochs_range, val_loss_, label="Val Loss",   color="#f43f5e")
axes[1].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.5, label="Fine-tune start")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/content/model/training_history.png", dpi=120, bbox_inches='tight')
plt.show()

print("Plot disimpan di: /content/model/training_history.png")


# ============================================================
# CELL 11: Simpan Class Indices
# ============================================================

class_indices = train_gen.class_indices
class_info_path = "/content/model/class_indices.json"
with open(class_info_path, "w") as f:
    json.dump(class_indices, f, indent=2)

print(f"Class indices: {class_indices}")
print(f"Disimpan di: {class_info_path}")

# Label mapping
label_map = {v: ("Organik" if k == "O" else "Anorganik") for k, v in class_indices.items()}
print(f"Label map: {label_map}")


# ============================================================
# CELL 12: Test Prediksi dengan Gambar Sample
# ============================================================

from tensorflow.keras.preprocessing import image
import random

# Ambil random gambar dari validation set
test_class = random.choice(['O', 'R'])
test_dir = os.path.join(DATASET_DIR, test_class)
test_files = os.listdir(test_dir)
test_file = random.choice(test_files)
test_path = os.path.join(test_dir, test_file)

# Load dan preprocess
img = image.load_img(test_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediksi
prediction = model.predict(img_array, verbose=0)
score = float(prediction[0][0])

if score >= 0.5:
    predicted_label = "Anorganik"
    confidence = score * 100
else:
    predicted_label = "Organik"
    confidence = (1 - score) * 100

actual_label = "Organik" if test_class == "O" else "Anorganik"

# Tampilkan
plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.axis('off')
plt.title(f"Prediksi: {predicted_label} ({confidence:.1f}%)\nAktual: {actual_label}", fontsize=12)
plt.show()

print(f"File: {test_file}")
print(f"Prediksi: {predicted_label}")
print(f"Confidence: {confidence:.2f}%")
print(f"Aktual: {actual_label}")
print(f"{'✅ BENAR' if predicted_label == actual_label else '❌ SALAH'}")


# ============================================================
# CELL 13: Download Model
# ============================================================

from google.colab import files

print("Downloading model...")
files.download('/content/model/waste_classifier.h5')
files.download('/content/model/class_indices.json')
files.download('/content/model/training_history.png')
print("Download selesai!")
