# ============================================================================
# STEP 2: DATASET PROCESSOR (OVERLAPPING WINDOWS) — v3
# PSL Translator - Final Year Project
#
# SAME as your working v2 preprocessor.
# ONLY CHANGE: orientation check skips 1080p_ mobile videos
#              (they are already upright, no rotation needed)
#
# INPUT:  dataset/ folder with .mp4/.MOV videos
# OUTPUT: data_keypoints_v3/ folder with .npy files
# ============================================================================

import cv2
import numpy as np
import os
import mediapipe as mp
import shutil

print("=" * 80)
print("🦴 STEP 2: DATASET PROCESSOR (v3 — Webcam + Mobile)")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = os.path.join(os.getcwd(), "dataset")
OUTPUT_PATH = os.path.join(os.getcwd(), "data_keypoints_v3")
SEQUENCE_LENGTH = 30
WINDOW_STRIDE = 5
TARGET_AUG_PER_SAMPLE = 4

# ============================================================================
# SETUP
# ============================================================================
if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)
os.makedirs(OUTPUT_PATH)

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ============================================================================
# 1. FEATURE EXTRACTION
# ============================================================================
def extract_features(results):
    if results.pose_landmarks:
        res = results.pose_landmarks.landmark
        upper_body = np.array([
            [res[11].x, res[11].y, res[11].z],
            [res[12].x, res[12].y, res[12].z],
            [res[13].x, res[13].y, res[13].z],
            [res[14].x, res[14].y, res[14].z],
            [res[15].x, res[15].y, res[15].z],
            [res[16].x, res[16].y, res[16].z],
        ]).flatten()
        anchors = np.array([
            [res[11].x, res[11].y, res[11].z],
            [res[12].x, res[12].y, res[12].z],
            [res[23].x, res[23].y, res[23].z],
            [res[24].x, res[24].y, res[24].z],
        ])
    else:
        upper_body = np.zeros(18)
        anchors = np.zeros((4, 3))

    lh = (
        np.array([[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(63)
    )
    rh = (
        np.array([[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(63)
    )
    return upper_body, lh, rh, anchors


# ============================================================================
# 2. NORMALIZATION
# ============================================================================
def normalize_frame(pose, lh, rh, anchors):
    l_sh, r_sh = anchors[0], anchors[1]
    if np.sum(l_sh) == 0 or np.sum(r_sh) == 0:
        return None

    center = (l_sh + r_sh) / 2
    mid_shoulder = (l_sh + r_sh) / 2
    l_hip, r_hip = anchors[2], anchors[3]

    if np.sum(l_hip) != 0 and np.sum(r_hip) != 0:
        mid_hip = (l_hip + r_hip) / 2
        scale = np.linalg.norm(mid_shoulder - mid_hip)
    else:
        scale = np.linalg.norm(l_sh - r_sh) * 1.5

    if scale < 0.1:
        scale = 1

    def norm(data):
        if len(data) == 0:
            return data
        reshaped = data.reshape(-1, 3)
        mask = np.any(reshaped != 0, axis=1)
        reshaped[mask] = (reshaped[mask] - center) / scale
        return reshaped.flatten()

    return np.concatenate([norm(pose), norm(lh), norm(rh)])


# ============================================================================
# 3. ORIENTATION CHECK
#    - SKIP for webcam videos (user_webcam_*) — already correct
#    - SKIP for mobile videos (1080p_*) — already upright
#    - SKIP for idle videos (_idle_)
#    - ONLY CHECK for original dataset videos (MVI_*.MOV etc.)
# ============================================================================
def check_orientation(frame):
    with mp_holistic.Holistic(static_image_mode=True) as h:
        res = h.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            l_y = res.pose_landmarks.landmark[11].y
            r_y = res.pose_landmarks.landmark[12].y
            l_x = res.pose_landmarks.landmark[11].x
            r_x = res.pose_landmarks.landmark[12].x
            if abs(l_y - r_y) > abs(l_x - r_x):
                return True
    return False


def needs_orientation_check(filename, is_idle):
    """
    Returns True only for original dataset videos (MVI_*.MOV etc.)
    Webcam and mobile videos are already correctly oriented.
    """
    name_lower = filename.lower()

    # These are already correct — no orientation check needed
    if is_idle:
        return False
    if name_lower.startswith("user_webcam"):
        return False
    if name_lower.startswith("1080p_"):
        return False

    # Original dataset videos — might need rotation
    return True


# ============================================================================
# 4. AUGMENTATION
# ============================================================================
def augment_mirror(sequence):
    seq = sequence.reshape(sequence.shape[0], -1, 3)
    seq[:, :, 0] *= -1
    pose = seq[:, :6, :]
    pose = pose[:, [1, 0, 3, 2, 5, 4], :]
    lh = seq[:, 6:27, :]
    rh = seq[:, 27:48, :]
    return np.concatenate([pose, rh, lh], axis=1).reshape(sequence.shape[0], -1)


def augment_affine(sequence):
    angle = np.random.uniform(-10, 10)
    scale = np.random.uniform(0.9, 1.1)
    rad = np.radians(angle)
    c, s = np.cos(rad), np.sin(rad)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    seq = sequence.reshape(-1, 3)
    aug_seq = np.dot(seq, rot) * scale

    x_out = np.abs(aug_seq[:, 0]) > 1.5
    y_out = np.abs(aug_seq[:, 1]) > 1.5
    out_mask = x_out | y_out
    aug_seq[out_mask] = 0

    return aug_seq.reshape(sequence.shape)


# ============================================================================
# 5. VIDEO → ALL FRAMES
# ============================================================================
def extract_all_frames(video_path, vid_name, is_idle=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Orientation check — only for original dataset videos
    rotate_90 = False
    if needs_orientation_check(vid_name, is_idle):
        ret, check_frame = cap.read()
        if not ret:
            cap.release()
            return None
        rotate_90 = check_orientation(check_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    all_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if rotate_90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        p, l, r, a = extract_features(results)
        norm = normalize_frame(p, l, r, a)

        if norm is not None:
            all_frames.append(norm)
        else:
            all_frames.append(np.zeros(144))

    cap.release()

    if len(all_frames) < SEQUENCE_LENGTH:
        return None

    return np.array(all_frames)


# ============================================================================
# 6. CREATE OVERLAPPING WINDOWS
# ============================================================================
def create_windows(all_frames):
    windows = []
    total_frames = len(all_frames)

    for start in range(0, total_frames - SEQUENCE_LENGTH + 1, WINDOW_STRIDE):
        window = all_frames[start : start + SEQUENCE_LENGTH]
        windows.append(window)

    return windows


# ============================================================================
# 7. MAIN PROCESSING LOOP
# ============================================================================
classes = sorted(
    [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
)

print(f"\n🚀 Processing {len(classes)} classes (including _idle_)")
print(f"   Window size: {SEQUENCE_LENGTH} frames")
print(f"   Window stride: {WINDOW_STRIDE} frames")
print(f"   Augmentations per sample: base + mirror + {TARGET_AUG_PER_SAMPLE} affine each")
print("-" * 80)

total_samples = 0
source_stats = {"webcam": 0, "mobile": 0, "original": 0, "idle": 0}

for cls in classes:
    is_idle = cls == "_idle_"
    src = os.path.join(DATASET_PATH, cls)
    dst = os.path.join(OUTPUT_PATH, cls)
    os.makedirs(dst, exist_ok=True)

    videos = [
        f
        for f in os.listdir(src)
        if f.lower().endswith((".mp4", ".mov", ".avi"))
    ]

    class_count = 0
    sample_id = 0

    print(f"\n📂 {cls} ({len(videos)} videos) {'[IDLE CLASS]' if is_idle else ''}")

    for vid_name in videos:
        vid_path = os.path.join(src, vid_name)

        # Track source
        name_lower = vid_name.lower()
        if is_idle:
            source_stats["idle"] += 1
            src_tag = "IDL"
        elif name_lower.startswith("user_webcam"):
            source_stats["webcam"] += 1
            src_tag = "WEB"
        elif name_lower.startswith("1080p_"):
            source_stats["mobile"] += 1
            src_tag = "MOB"
        else:
            source_stats["original"] += 1
            src_tag = "ORI"

        # Step A: Extract ALL frames
        all_frames = extract_all_frames(vid_path, vid_name, is_idle=is_idle)
        if all_frames is None:
            print(f"   ⚠️  Skipped {vid_name} (too short or unreadable)")
            continue

        # Step B: Create overlapping windows
        windows = create_windows(all_frames)

        print(f"   📹 [{src_tag}] {vid_name}: {len(all_frames)} frames → {len(windows)} windows")

        # Step C: Save each window + augmentations
        for window in windows:
            # --- Base ---
            np.save(os.path.join(dst, f"{sample_id}_base.npy"), window)
            class_count += 1

            # --- Mirror ---
            mirror = augment_mirror(window.copy())
            np.save(os.path.join(dst, f"{sample_id}_mir.npy"), mirror)
            class_count += 1

            # --- Affine augmentations on base ---
            for j in range(TARGET_AUG_PER_SAMPLE):
                aug = augment_affine(window.copy())
                np.save(os.path.join(dst, f"{sample_id}_aug_{j}.npy"), aug)
                class_count += 1

            # --- Affine augmentations on mirror ---
            for j in range(TARGET_AUG_PER_SAMPLE):
                aug = augment_affine(mirror.copy())
                np.save(os.path.join(dst, f"{sample_id}_mir_aug_{j}.npy"), aug)
                class_count += 1

            sample_id += 1

    print(f"   ✅ {cls}: {class_count} total samples")
    total_samples += class_count

# ============================================================================
# 8. SUMMARY
# ============================================================================
holistic.close()

print("\n" + "=" * 80)
print("🎉 DATASET PROCESSING COMPLETE")
print("=" * 80)
print(f"   Total classes: {len(classes)} (including _idle_)")
print(f"   Total samples: {total_samples}")
print(f"   Output folder: {OUTPUT_PATH}")
print(f"   Each sample shape: ({SEQUENCE_LENGTH}, 144)")
print(f"\n   Source breakdown:")
print(f"     Original dataset videos: {source_stats['original']}")
print(f"     Webcam videos:           {source_stats['webcam']}")
print(f"     Mobile 1080p videos:     {source_stats['mobile']}")
print(f"     Idle videos:             {source_stats['idle']}")
print(f"\n   Next step: Run step3_verify_data.py")