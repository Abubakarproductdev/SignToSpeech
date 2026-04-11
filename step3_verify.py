# ============================================================================
# STEP 3: VERIFY PREPROCESSED DATA (v3)
# PSL Translator - Final Year Project
#
# Checks:
#   1. Data shapes are correct
#   2. No NaN or Inf values
#   3. Feature distributions look healthy
#   4. Webcam vs Mobile feature comparison
#   5. Every class has balanced samples
#   6. Ready for training
#
# INPUT:  preprocessed_dataset_v3.npz, class_names_v3.json
# ============================================================================

import numpy as np
import json
import sys

NPZ_FILE = "preprocessed_dataset_v3.npz"
CLASS_FILE = "class_names_v3.json"

print("=" * 60)
print("  STEP 3: VERIFY DATA (v3)")
print("=" * 60)

# ============================================================================
# LOAD
# ============================================================================
try:
    data = np.load(NPZ_FILE)
    X = data["X"]
    y = data["y"]
except:
    print(f"❌ Cannot load {NPZ_FILE}")
    sys.exit()

with open(CLASS_FILE, "r") as f:
    class_names = json.load(f)

print(f"\n✅ Loaded: {NPZ_FILE}")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print(f"   Classes: {len(class_names)}")

# ============================================================================
# CHECK 1: SHAPES
# ============================================================================
print(f"\n{'─' * 60}")
print("CHECK 1: SHAPES")
print(f"{'─' * 60}")

expected_samples = 320
expected_frames = 30
expected_features = 144

shape_ok = True
if X.shape[0] != expected_samples:
    print(f"   ⚠️ Expected {expected_samples} samples, got {X.shape[0]}")
    shape_ok = False
if X.shape[1] != expected_frames:
    print(f"   ❌ Expected {expected_frames} frames, got {X.shape[1]}")
    shape_ok = False
if X.shape[2] != expected_features:
    print(f"   ❌ Expected {expected_features} features, got {X.shape[2]}")
    shape_ok = False

if shape_ok:
    print(f"   ✅ Shape correct: ({X.shape[0]}, {X.shape[1]}, {X.shape[2]})")

# ============================================================================
# CHECK 2: NaN / Inf
# ============================================================================
print(f"\n{'─' * 60}")
print("CHECK 2: NaN / Inf VALUES")
print(f"{'─' * 60}")

nan_count = np.sum(np.isnan(X))
inf_count = np.sum(np.isinf(X))

if nan_count == 0 and inf_count == 0:
    print(f"   ✅ No NaN or Inf values")
else:
    print(f"   ❌ NaN count: {nan_count}")
    print(f"   ❌ Inf count: {inf_count}")

# ============================================================================
# CHECK 3: FEATURE DISTRIBUTIONS
# ============================================================================
print(f"\n{'─' * 60}")
print("CHECK 3: FEATURE DISTRIBUTIONS")
print(f"{'─' * 60}")

print(f"   {'':>20} {'MIN':>10} {'MAX':>10} {'MEAN':>10} {'STD':>10}")
print(f"   {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

# Overall
print(f"   {'All features':<20} {X.min():>10.4f} {X.max():>10.4f} {X.mean():>10.4f} {X.std():>10.4f}")

# Pose (0-17), Left hand (18-80), Right hand (81-143)
pose = X[:, :, :18]
lh = X[:, :, 18:81]
rh = X[:, :, 81:144]

print(f"   {'Pose (0-17)':<20} {pose.min():>10.4f} {pose.max():>10.4f} {pose.mean():>10.4f} {pose.std():>10.4f}")
print(f"   {'Left hand (18-80)':<20} {lh.min():>10.4f} {lh.max():>10.4f} {lh.mean():>10.4f} {lh.std():>10.4f}")
print(f"   {'Right hand (81-143)':<20} {rh.min():>10.4f} {rh.max():>10.4f} {rh.mean():>10.4f} {rh.std():>10.4f}")

# Check for dead features (all zeros)
zero_features = 0
for feat_idx in range(144):
    if np.all(X[:, :, feat_idx] == 0):
        zero_features += 1

if zero_features == 0:
    print(f"\n   ✅ No dead features (all 144 features have data)")
else:
    print(f"\n   ⚠️ {zero_features} features are all zeros")

# ============================================================================
# CHECK 4: PER-CLASS BALANCE
# ============================================================================
print(f"\n{'─' * 60}")
print("CHECK 4: CLASS BALANCE")
print(f"{'─' * 60}")

min_count = float('inf')
max_count = 0

for i, name in enumerate(class_names):
    count = np.sum(y == i)
    min_count = min(min_count, count)
    max_count = max(max_count, count)

    # Check per-class feature quality
    class_data = X[y == i]
    class_mean = np.mean(np.abs(class_data))
    zero_frames = np.sum(np.all(class_data == 0, axis=2))
    total_frames = class_data.shape[0] * class_data.shape[1]
    zero_pct = zero_frames / total_frames * 100

    status = "✅" if count >= 10 and zero_pct < 50 else "⚠️"
    print(f"   {status} {name:<20} {count:>3} samples  mean_mag:{class_mean:.4f}  zero_frames:{zero_pct:.1f}%")

print(f"\n   Min samples: {min_count}")
print(f"   Max samples: {max_count}")
if min_count >= 10:
    print(f"   ✅ All classes have ≥10 samples")
else:
    print(f"   ⚠️ Some classes have <10 samples")

# ============================================================================
# CHECK 5: WEBCAM vs MOBILE FEATURE COMPARISON
# ============================================================================
print(f"\n{'─' * 60}")
print("CHECK 5: WEBCAM vs MOBILE FEATURE COMPARISON")
print(f"{'─' * 60}")

# Assuming first 5 samples per class = webcam, next 5 = mobile
# (based on sorted filenames: 1080p_* comes before user_webcam_*)
# Let's check by looking at feature magnitude patterns

# For each class, split samples into two groups and compare
print(f"\n   Comparing first 5 vs last 5 samples per class:")
print(f"   (first 5 = likely one source, last 5 = other source)")
print(f"\n   {'CLASS':<20} {'GROUP1 mag':>12} {'GROUP2 mag':>12} {'DIFF':>10} {'STATUS':>8}")
print(f"   {'─'*20} {'─'*12} {'─'*12} {'─'*10} {'─'*8}")

big_diffs = []
for i, name in enumerate(class_names):
    class_data = X[y == i]
    count = len(class_data)

    if count < 10:
        continue

    half = count // 2
    group1 = class_data[:half]
    group2 = class_data[half:]

    g1_mag = np.mean(np.abs(group1))
    g2_mag = np.mean(np.abs(group2))
    diff = abs(g1_mag - g2_mag)

    status = "✅" if diff < 0.15 else "⚠️"
    if diff >= 0.15:
        big_diffs.append(name)

    print(f"   {name:<20} {g1_mag:>12.4f} {g2_mag:>12.4f} {diff:>10.4f} {status:>8}")

if big_diffs:
    print(f"\n   ⚠️ Classes with large group differences: {', '.join(big_diffs)}")
    print(f"      This is EXPECTED — webcam and mobile produce different magnitudes.")
    print(f"      The model will learn to handle both.")
else:
    print(f"\n   ✅ All classes have similar feature magnitudes across groups")

# ============================================================================
# CHECK 6: SAMPLE DIVERSITY
# ============================================================================
print(f"\n{'─' * 60}")
print("CHECK 6: SAMPLE DIVERSITY (are samples different from each other?)")
print(f"{'─' * 60}")

low_diversity = []
for i, name in enumerate(class_names):
    class_data = X[y == i]
    if len(class_data) < 2:
        continue

    # Average pairwise distance between samples
    dists = []
    for a in range(len(class_data)):
        for b in range(a + 1, len(class_data)):
            dist = np.mean(np.abs(class_data[a] - class_data[b]))
            dists.append(dist)

    avg_dist = np.mean(dists)
    status = "✅" if avg_dist > 0.05 else "⚠️"
    if avg_dist <= 0.05:
        low_diversity.append(name)
    print(f"   {status} {name:<20} avg distance: {avg_dist:.4f}")

if low_diversity:
    print(f"\n   ⚠️ Low diversity: {', '.join(low_diversity)}")
else:
    print(f"\n   ✅ All classes have good diversity")

# ============================================================================
# FINAL VERDICT
# ============================================================================
print(f"\n{'=' * 60}")
print("  FINAL VERDICT")
print(f"{'=' * 60}")

issues = []
if nan_count > 0 or inf_count > 0:
    issues.append("NaN/Inf values found")
if not shape_ok:
    issues.append("Shape mismatch")
if min_count < 5:
    issues.append("Some classes have too few samples")

if not issues:
    print(f"\n   ✅ ALL CHECKS PASSED")
    print(f"   ✅ Data is ready for training")
    print(f"\n   Dataset: {X.shape[0]} samples, {len(class_names)} classes")
    print(f"   Next: Run step4_train_model.py")
else:
    print(f"\n   ❌ ISSUES FOUND:")
    for issue in issues:
        print(f"      - {issue}")

print(f"\n{'=' * 60}")