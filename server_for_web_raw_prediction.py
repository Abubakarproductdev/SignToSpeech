# This is the final corrected version for web, it is based on the raw frames prdictions from the ai. the problem is mobile and the webs frames differ from each other, so it is not giving best results on the mobile phone

from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import json
import os
import tempfile
from collections import deque

app = Flask(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "psl_model_v3.h5"
CLASS_FILE = "class_names_v3.json"
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.70

# ============================================================================
# LOAD
# ============================================================================
print("=" * 60)
print("  PSL TRANSLATOR SERVER (v3 — FINAL)")
print("=" * 60)

model = keras.models.load_model(MODEL_PATH)
with open(CLASS_FILE, "r") as f:
    class_names = json.load(f)

print(f"   ✅ Model loaded: {MODEL_PATH}")
print(f"   ✅ Classes: {len(class_names)}")

mp_holistic = mp.solutions.holistic


# ============================================================================
# FEATURE EXTRACTION (Same as step5)
# ============================================================================
def extract_features(results):
    if results.pose_landmarks:
        res = results.pose_landmarks.landmark
        upper_body = np.array(
            [
                [res[11].x, res[11].y, res[11].z],
                [res[12].x, res[12].y, res[12].z],
                [res[13].x, res[13].y, res[13].z],
                [res[14].x, res[14].y, res[14].z],
                [res[15].x, res[15].y, res[15].z],
                [res[16].x, res[16].y, res[16].z],
            ]
        ).flatten()
        anchors = np.array(
            [
                [res[11].x, res[11].y, res[11].z],
                [res[12].x, res[12].y, res[12].z],
                [res[23].x, res[23].y, res[23].z],
                [res[24].x, res[24].y, res[24].z],
            ]
        )
    else:
        upper_body = np.zeros(18)
        anchors = np.zeros((4, 3))

    lh = (
        np.array(
            [[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(63)
    )
    rh = (
        np.array(
            [[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(63)
    )
    return upper_body, lh, rh, anchors


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
# PROCESS VIDEO — EXACT SAME LOGIC AS STEP 5 (LIVE WEBCAM)
# ============================================================================
def process_video(video_path):
    """
    Processes video using the EXACT same frame-by-frame approach
    as step5 live webcam test (which works perfectly).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ""

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   📹 Video: {w}x{h} @ {fps:.0f} FPS ({total} frames)")

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── Same variables as step5 ──
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prediction_history = []
    current_prediction = ""
    current_confidence = 0.0

    frame_count = 0
    hands_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── Process frame (same as step5) ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        hands_visible = (
            results.left_hand_landmarks is not None
            or results.right_hand_landmarks is not None
        )
        if hands_visible:
            hands_count += 1

        # ── Extract features (same as step5) ──
        p, l, r, a = extract_features(results)
        norm = normalize_frame(p, l, r, a)

        if norm is not None:
            frame_buffer.append(norm)
        else:
            frame_buffer.append(np.zeros(144))

        # ── Predict when buffer is full (same as step5) ──
        if len(frame_buffer) == SEQUENCE_LENGTH and hands_visible:
            sequence = np.array(list(frame_buffer))
            inp = np.expand_dims(sequence, axis=0)
            probs = model.predict(inp, verbose=0)[0]
            idx = np.argmax(probs)
            conf = float(probs[idx])
            pred = class_names[idx]

            if pred != "_idle_" and conf >= CONFIDENCE_THRESHOLD:
                prediction_history.append(pred)
                current_prediction = pred
                current_confidence = conf

        frame_count += 1

    cap.release()
    holistic.close()

    print(f"   ✋ Hands: {hands_count}/{frame_count} frames")
    print(f"   📊 Raw predictions: {prediction_history}")

    if not prediction_history:
        return ""

    # ── Deduplicate consecutive same words ──
    final_words = []
    last_word = ""
    word_counts = {}

    for pred in prediction_history:
        if pred not in word_counts:
            word_counts[pred] = 0
        word_counts[pred] += 1

    # Method: Take the most common predictions, in order of first appearance
    # Remove noise (words that appear less than 3 times)
    min_appearances = max(2, len(prediction_history) * 0.05)

    seen_order = []
    for pred in prediction_history:
        if pred not in seen_order:
            seen_order.append(pred)

    for word in seen_order:
        if word_counts[word] >= min_appearances:
            if not final_words or final_words[-1] != word:
                final_words.append(word)

    print(f"   📊 Word counts: {word_counts}")
    print(f"   📊 Min appearances needed: {min_appearances:.0f}")
    print(f"   📝 Final words: {final_words}")

    sentence = " ".join(final_words)
    return sentence


# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.route("/predict_sentence", methods=["POST"])
def predict_sentence():
    print(f"\n{'━' * 60}")
    print(f"   📥 RECEIVED VIDEO FROM MOBILE")
    print(f"{'━' * 60}")

    if "video" not in request.files:
        return jsonify({"error": "No video file received"}), 400

    video_file = request.files["video"]

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "received_video.mp4")
    video_file.save(temp_path)

    file_size = os.path.getsize(temp_path) / 1024
    print(f"   💾 File size: {file_size:.1f} KB")

    sentence = process_video(temp_path)

    # try:
    #     # os.remove(temp_path)
    #     os.rmdir(temp_dir)

    # except:
    #     pass

    if sentence:
        print(f"   ✅ RESULT: {sentence}")
        return jsonify({"sentence": sentence})
    else:
        print(f"   ❌ No signs detected")
        return jsonify({"sentence": "", "error": "No signs detected"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model": MODEL_PATH,
            "classes": len(class_names),
        }
    )


# ============================================================================
# START
# ============================================================================
if __name__ == "__main__":
    print(f"\n   🚀 http://0.0.0.0:5000")
    print(f"   📱 POST /predict_sentence")
    print(f"   ❤️  GET /health")
    app.run(host="0.0.0.0", port=5000, debug=False)
