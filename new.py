# ============================================================================
# CONVERT MODEL TO TFLITE (GRU-compatible)
# ============================================================================
import tensorflow as tf
import numpy as np
import json
import os

MODEL_PATH = "psl_model_v3.h5"
TFLITE_PATH = "psl_model_v3.tflite"
CLASS_FILE = "class_names_v3.json"

print("=" * 60)
print("  CONVERT MODEL TO TFLITE (GRU)")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)
print(f"   ✅ Loaded: {MODEL_PATH}")
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# ✅ REQUIRED for GRU/LSTM models
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False

# Optional optimization (still safe with select ops)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

tflite_size = os.path.getsize(TFLITE_PATH) / 1024
h5_size = os.path.getsize(MODEL_PATH) / 1024

print(f"\n   ✅ Saved: {TFLITE_PATH}")
print(f"   Original size: {h5_size:.0f} KB")
print(f"   TFLite size:   {tflite_size:.0f} KB")

print(f"\n   ✅ Conversion complete (GRU compatible)")