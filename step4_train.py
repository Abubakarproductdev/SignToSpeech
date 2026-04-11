# ============================================================================
# STEP 4: TRAIN MODEL (v3 — Mixed Webcam + Mobile)
# PSL Translator - Final Year Project
#
# Same architecture as v2 (LSTM + Dense)
# Same training strategy
# Trains on mixed webcam + mobile data
#
# INPUT:  preprocessed_dataset_v3.npz, class_names_v3.json
# OUTPUT: psl_model_v3.h5
# ============================================================================

import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
NPZ_FILE = "preprocessed_dataset_v3.npz"
CLASS_FILE = "class_names_v3.json"
MODEL_OUTPUT = "psl_model_v3.h5"

EPOCHS = 100
BATCH_SIZE = 16
PATIENCE = 15  # early stopping patience
VALIDATION_SPLIT = 0.2

print("=" * 60)
print("  STEP 4: TRAIN MODEL (v3 — Mixed Sources)")
print("=" * 60)

# ============================================================================
# LOAD DATA
# ============================================================================
data = np.load(NPZ_FILE)
X = data["X"]
y = data["y"]

with open(CLASS_FILE, "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

print(f"\n   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print(f"   Classes: {num_classes}")

# ============================================================================
# TRAIN/VAL SPLIT (stratified — equal webcam+mobile in both sets)
# ============================================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=VALIDATION_SPLIT,
    random_state=42,
    stratify=y,
)

print(f"\n   Train: {X_train.shape[0]} samples")
print(f"   Val:   {X_val.shape[0]} samples")

# ============================================================================
# CLASS WEIGHTS (handles _idle_ having 20 samples vs 10 for others)
# ============================================================================
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
)
class_weights = {i: w for i, w in enumerate(class_weights_array)}

print(f"\n   Class weights computed (balanced)")

# ============================================================================
# DATA AUGMENTATION (noise injection for robustness)
# ============================================================================
def augment_data(X_data, y_data, noise_levels=[0.01, 0.02]):
    """
    Add Gaussian noise copies to increase training data.
    This helps the model generalize across webcam/mobile differences.
    """
    augmented_X = [X_data]
    augmented_y = [y_data]

    for noise in noise_levels:
        noisy = X_data + np.random.normal(0, noise, X_data.shape)
        augmented_X.append(noisy.astype(np.float32))
        augmented_y.append(y_data)

    return np.concatenate(augmented_X), np.concatenate(augmented_y)

X_train_aug, y_train_aug = augment_data(X_train, y_train)
print(f"\n   After augmentation:")
print(f"   Train: {X_train_aug.shape[0]} samples (original + 2 noise levels)")

# ============================================================================
# BUILD MODEL (Same architecture as v2)
# ============================================================================
model = keras.Sequential([
    layers.Input(shape=(X.shape[1], X.shape[2])),  # (30, 144)

    # LSTM layers
    layers.LSTM(128, return_sequences=True, dropout=0.3),
    layers.LSTM(64, return_sequences=False, dropout=0.3),

    # Dense layers
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),

    # Output
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ============================================================================
# CALLBACKS
# ============================================================================
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=0.00001,
        verbose=1,
    ),
]

# ============================================================================
# TRAIN
# ============================================================================
print(f"\n{'=' * 60}")
print(f"  TRAINING")
print(f"{'=' * 60}")
print(f"  Epochs: {EPOCHS} (early stopping patience: {PATIENCE})")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Training samples: {len(X_train_aug)}")
print(f"  Validation samples: {len(X_val)}")

history = model.fit(
    X_train_aug, y_train_aug,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

# ============================================================================
# EVALUATE
# ============================================================================
print(f"\n{'=' * 60}")
print(f"  EVALUATION")
print(f"{'=' * 60}")

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

print(f"\n   Train accuracy: {train_acc:.1%}")
print(f"   Val accuracy:   {val_acc:.1%}")
print(f"   Train loss:     {train_loss:.4f}")
print(f"   Val loss:       {val_loss:.4f}")

# Overfitting check
gap = train_acc - val_acc
if gap > 0.15:
    print(f"\n   ⚠️ Overfitting detected (gap: {gap:.1%})")
elif gap < 0.05:
    print(f"\n   ✅ Good generalization (gap: {gap:.1%})")
else:
    print(f"\n   ⚠️ Slight overfitting (gap: {gap:.1%})")

# ============================================================================
# PER-CLASS ACCURACY ON VALIDATION SET
# ============================================================================
print(f"\n{'─' * 60}")
print(f"  PER-CLASS VALIDATION ACCURACY")
print(f"{'─' * 60}")

val_preds = model.predict(X_val, verbose=0)
val_pred_classes = np.argmax(val_preds, axis=1)

for i, name in enumerate(class_names):
    mask = y_val == i
    if np.sum(mask) == 0:
        continue
    correct = np.sum(val_pred_classes[mask] == i)
    total = np.sum(mask)
    acc = correct / total
    status = "✅" if acc >= 0.5 else "❌"
    print(f"   {status} {name:<20} {correct}/{total}  ({acc:.0%})")

# ============================================================================
# SAVE
# ============================================================================
model.save(MODEL_OUTPUT)
print(f"\n{'=' * 60}")
print(f"  ✅ Model saved: {MODEL_OUTPUT}")
print(f"{'=' * 60}")

# Also save training history
best_epoch = np.argmax(history.history["val_accuracy"]) + 1
best_val_acc = max(history.history["val_accuracy"])

print(f"\n   Best epoch: {best_epoch}")
print(f"   Best val accuracy: {best_val_acc:.1%}")
print(f"\n   Next: Run step5_test_model.py")
print(f"{'=' * 60}")