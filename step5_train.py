# ============================================================================
# STEP 4: TRAIN MODEL (v3 — Webcam + Mobile + Original)
# PSL Translator - Final Year Project
#
# IDENTICAL to your working v2 training script.
# ONLY CHANGES:
#   1. Loads from data_keypoints_v3/ (has mobile data too)
#   2. Saves as psl_model_v3.h5 (keeps v2 as backup)
#   3. Saves as class_names_v3.json
#
# ARCHITECTURE: Same GRU as v2 (proven to work)
# INPUT: (30 frames, 144 features)
# ============================================================================

import os
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

print("=" * 80)
print("🧠 STEP 4: MODEL TRAINING (v3 — Mixed Sources)")
print("=" * 80)

# ============================================================================
# CONFIGURATION — ONLY THESE 3 LINES CHANGED
# ============================================================================
DATA_PATH = os.path.join(os.getcwd(), "data_keypoints_v3")
MODEL_PATH = "psl_model_v3.h5"
CLASS_FILE = "class_names_v3.json"
BATCH_SIZE = 32
EPOCHS = 30

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1️⃣  Loading data...")

if not os.path.exists(DATA_PATH):
    print("❌ ERROR: data_keypoints_v3 folder not found.")
    exit()

classes = sorted(
    [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
)
label_map = {label: num for num, label in enumerate(classes)}

print(f"   Classes ({len(classes)}): {classes}")

X, y = [], []

for cls in classes:
    cls_path = os.path.join(DATA_PATH, cls)
    files = [f for f in os.listdir(cls_path) if f.endswith(".npy")]

    loaded = 0
    for f in files:
        try:
            data = np.load(os.path.join(cls_path, f))
            if data.shape == (30, 144):
                X.append(data)
                y.append(label_map[cls])
                loaded += 1
        except:
            pass

    print(f"   ✅ {cls}: {loaded} samples loaded")

X = np.array(X)
y_raw = np.array(y)
y_cat = to_categorical(y_raw, num_classes=len(classes))

print(f"\n   Total samples: {X.shape[0]}")
print(f"   Input shape: {X.shape}")

if len(X) == 0:
    print("❌ ERROR: No data loaded.")
    exit()

# ============================================================================
# 2. COMPUTE CLASS WEIGHTS
# ============================================================================
print("\n2️⃣  Computing class weights...")

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_raw),
    y=y_raw,
)
class_weights = {i: w for i, w in enumerate(weights)}

print("   Class weights (higher = model pays more attention):")
for i, cls in enumerate(classes):
    print(f"   {cls:<25} weight: {class_weights[i]:.3f}")

# ============================================================================
# 3. SPLIT DATA
# ============================================================================
print("\n3️⃣  Splitting data (85% train, 15% validation)...")

X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat,
    test_size=0.15,
    random_state=42,
    stratify=y_raw,
)

print(f"   Train: {X_train.shape[0]} samples")
print(f"   Val:   {X_val.shape[0]} samples")

# ============================================================================
# 4. BUILD MODEL (Same GRU architecture as v2)
# ============================================================================
print("\n4️⃣  Building model...")

model = keras.Sequential([
    layers.Input(shape=(30, 144)),

    # GRU Layer 1
    layers.GRU(128, return_sequences=True, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    # GRU Layer 2
    layers.GRU(64, return_sequences=False, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    # Classifier
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(32, activation="relu"),
    layers.Dense(len(classes), activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005, clipnorm=0.5),
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

model.summary()

# ============================================================================
# 5. TRAIN
# ============================================================================
print("\n5️⃣  Training...")

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=[
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
        ),
    ],
)

# ============================================================================
# 6. PER-CLASS ACCURACY
# ============================================================================
print("\n6️⃣  Per-class accuracy on validation set...")

y_pred = model.predict(X_val, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

print(f"\n   {'CLASS':<25} {'CORRECT':>8} {'TOTAL':>8} {'ACCURACY':>10}")
print(f"   {'─'*25} {'─'*8} {'─'*8} {'─'*10}")

weak_classes = []

for i, cls in enumerate(classes):
    mask = y_true_classes == i
    total = np.sum(mask)
    if total == 0:
        continue
    correct = np.sum(y_pred_classes[mask] == i)
    acc = correct / total * 100

    status = "✅" if acc >= 70 else "⚠️" if acc >= 50 else "❌"
    print(f"   {status} {cls:<23} {correct:>8} {total:>8} {acc:>9.1f}%")

    if acc < 70:
        weak_classes.append((cls, acc))

if weak_classes:
    print(f"\n   ⚠️  {len(weak_classes)} classes below 70% accuracy:")
    for cls, acc in weak_classes:
        print(f"      → {cls}: {acc:.1f}% (consider recording more videos)")
else:
    print("\n   ✅ All classes above 70% accuracy!")

# ============================================================================
# 7. SAVE MODEL AND CLASS NAMES
# ============================================================================
print("\n7️⃣  Saving...")

model.save(MODEL_PATH)
print(f"   ✅ Model saved: {MODEL_PATH}")

with open(CLASS_FILE, "w") as f:
    json.dump(classes, f, indent=2)
print(f"   ✅ Classes saved: {CLASS_FILE}")

# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================
best_val_acc = max(history.history["val_categorical_accuracy"])
best_val_loss = min(history.history["val_loss"])
total_epochs = len(history.history["val_loss"])

print("\n" + "=" * 80)
print("🎉 TRAINING COMPLETE")
print("=" * 80)
print(f"   Best validation accuracy: {best_val_acc * 100:.2f}%")
print(f"   Best validation loss:     {best_val_loss:.4f}")
print(f"   Epochs trained:           {total_epochs}")
print(f"   Model file:               {MODEL_PATH}")
print(f"   Class file:               {CLASS_FILE}")
print(f"\n   Next step: Run step5_test_model.py")