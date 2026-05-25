"""
train_model.py
Script training model klasifikasi sampah: Organic vs Recyclable
MobileNetV2 Transfer Learning — Fixed Version
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-GUI backend, no plt.show() needed
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────

# Path langsung ke folder yang berisi O/ dan R/
DATASET_DIR  = r"C:\Users\Administrator\Downloads\archive (1)\DATASET\TRAIN"
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH   = os.path.join(MODEL_DIR, "waste_classifier.h5")

IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
EPOCHS_P1    = 15   # Phase 1: frozen base
EPOCHS_P2    = 10   # Phase 2: fine-tune
VAL_SPLIT    = 0.2
LR_P1        = 1e-3
LR_P2        = 1e-4

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# VALIDASI DATASET
# ─────────────────────────────────────────────
print("=" * 60)
print("  SMART WASTE CLASSIFIER — Training")
print("=" * 60)

# Pastikan hanya folder O dan R yang dipakai
valid_classes = ['O', 'R']
found = [d for d in os.listdir(DATASET_DIR)
         if os.path.isdir(os.path.join(DATASET_DIR, d)) and d in valid_classes]

print(f"\n  Dataset dir : {DATASET_DIR}")
print(f"  Kelas valid : {found}")

if len(found) != 2:
    raise ValueError(f"Harus ada folder O dan R di {DATASET_DIR}. Ditemukan: {found}")

# Hitung jumlah gambar
for cls in valid_classes:
    cls_path = os.path.join(DATASET_DIR, cls)
    n = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    print(f"  {cls} ({['Organic','Recyclable'][valid_classes.index(cls)]}): {n} gambar")

# ─────────────────────────────────────────────
# DATA GENERATOR — hanya load folder O dan R
# ─────────────────────────────────────────────

# Buat temporary directory structure yang bersih
# Gunakan classes parameter untuk filter hanya O dan R
print("\n[1/5] Menyiapkan data generator...")

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
    class_mode="binary",       # binary: 0 atau 1
    classes=valid_classes,     # HANYA O dan R, abaikan folder lain
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

# class_indices: {'O': 0, 'R': 1}
class_indices = train_gen.class_indices
print(f"\n  Class mapping : {class_indices}")
print(f"  Train samples : {train_gen.samples}")
print(f"  Val samples   : {val_gen.samples}")

# Simpan mapping
label_map = {v: ("Organic" if k == "O" else "Recyclable") for k, v in class_indices.items()}
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(class_indices, f, indent=2)
print(f"  Label map     : {label_map}")

# ─────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────
print("\n[2/5] Membangun model MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # Freeze semua layer base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)
output = Dense(1, activation="sigmoid")(x)  # binary output

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_P1),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

trainable_params = sum([np.prod(w.shape) for w in model.trainable_weights])
print(f"  Trainable params: {trainable_params:,}")

# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
callbacks_p1 = [
    ModelCheckpoint(
        MODEL_PATH,
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

# ─────────────────────────────────────────────
# PHASE 1: TRAINING (frozen base)
# ─────────────────────────────────────────────
print(f"\n[3/5] Phase 1 — Training head (base frozen, {EPOCHS_P1} epochs)...")

steps_per_epoch  = train_gen.samples // BATCH_SIZE
validation_steps = val_gen.samples   // BATCH_SIZE

history1 = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS_P1,
    callbacks=callbacks_p1,
    verbose=1,
)

# ─────────────────────────────────────────────
# PHASE 2: FINE-TUNING (unfreeze top layers)
# ─────────────────────────────────────────────
print(f"\n[4/5] Phase 2 — Fine-tuning (unfreeze top 40 layers, {EPOCHS_P2} epochs)...")

base_model.trainable = True
# Freeze semua kecuali 40 layer terakhir
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_P2),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks_p2 = [
    ModelCheckpoint(
        MODEL_PATH,
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

history2 = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS_P2,
    callbacks=callbacks_p2,
    verbose=1,
)

# ─────────────────────────────────────────────
# EVALUASI FINAL
# ─────────────────────────────────────────────
print("\n[5/5] Evaluasi model...")

val_gen.reset()
val_loss, val_acc = model.evaluate(val_gen, steps=validation_steps, verbose=0)

print(f"\n  ✅ Final Validation Accuracy : {val_acc * 100:.2f}%")
print(f"  ✅ Final Validation Loss     : {val_loss:.4f}")
print(f"\n  Model disimpan di: {MODEL_PATH}")

# ─────────────────────────────────────────────
# PLOT TRAINING HISTORY
# ─────────────────────────────────────────────
acc      = history1.history["accuracy"]      + history2.history["accuracy"]
val_acc_ = history1.history["val_accuracy"]  + history2.history["val_accuracy"]
loss_    = history1.history["loss"]          + history2.history["loss"]
val_loss_= history1.history["val_loss"]      + history2.history["val_loss"]

epochs_range = range(1, len(acc) + 1)
phase1_end   = len(history1.history["accuracy"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Smart Waste Classifier — Training History", fontsize=14, fontweight='bold')

# Accuracy
axes[0].plot(epochs_range, acc,      label="Train Accuracy",      color="#4ade80")
axes[0].plot(epochs_range, val_acc_, label="Val Accuracy",        color="#22d3ee")
axes[0].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.5, label="Fine-tune start")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1])

# Loss
axes[1].plot(epochs_range, loss_,     label="Train Loss",   color="#f97316")
axes[1].plot(epochs_range, val_loss_, label="Val Loss",     color="#f43f5e")
axes[1].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.5, label="Fine-tune start")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, "training_history.png")
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Plot disimpan di: {plot_path}")

print("\n" + "=" * 60)
print(f"  Training selesai!")
print(f"  Accuracy: {val_acc * 100:.2f}%")
print("=" * 60)
