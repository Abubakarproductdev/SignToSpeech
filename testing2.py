
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import json
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "psl_model_v3.h5"
CLASS_FILE = "class_names_v3.json"
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.70

print("=" * 60)
print("  STEP 5: LIVE TEST (v3)")
print("=" * 60)

# ============================================================================
# LOAD MODEL
# ============================================================================
model = keras.models.load_model(MODEL_PATH)
with open(CLASS_FILE, "r") as f:
    class_names = json.load(f)

print(f"   ✅ Model loaded: {MODEL_PATH}")
print(f"   ✅ Classes: {len(class_names)}")

# ============================================================================
# MEDIAPIPE SETUP
# ============================================================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ============================================================================
# FEATURE EXTRACTION (Same as training)
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
# LIVE DETECTION
# ============================================================================
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
prediction_history = deque(maxlen=10)
current_prediction = ""
current_confidence = 0.0
hands_visible = False

print(f"\n   🎥 Webcam started")
print(f"   📊 Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
print(f"   Press Q to quit")
print(f"{'─' * 60}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    # Check hands
    hands_visible = (
        results.left_hand_landmarks is not None
        or results.right_hand_landmarks is not None
    )

    # Extract features
    p, l, r, a = extract_features(results)
    norm = normalize_frame(p, l, r, a)

    if norm is not None:
        frame_buffer.append(norm)
    else:
        frame_buffer.append(np.zeros(144))

    # Predict when buffer is full
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

    # Draw landmarks
    display = cv2.flip(frame, 1)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            display, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 80, 80), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 80, 80), thickness=1),
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            display, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2),
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            display, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2),
        )

    # UI overlay
    # Top bar
    cv2.rectangle(display, (0, 0), (640, 80), (0, 0, 0), -1)

    if current_prediction and current_confidence >= CONFIDENCE_THRESHOLD:
        color = (0, 255, 0) if current_confidence >= 0.90 else (0, 255, 255)
        cv2.putText(display, f"{current_prediction}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(display, f"{current_confidence:.0%}", (500, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    else:
        cv2.putText(display, "Sign a word...", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)

    # Bottom bar
    cv2.rectangle(display, (0, 440), (640, 480), (0, 0, 0), -1)

    hand_status = "Hands: VISIBLE" if hands_visible else "Hands: NOT DETECTED"
    hand_color = (0, 255, 0) if hands_visible else (0, 0, 255)
    cv2.putText(display, hand_status, (10, 468),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)

    buffer_pct = len(frame_buffer) / SEQUENCE_LENGTH * 100
    cv2.putText(display, f"Buffer: {buffer_pct:.0f}%", (300, 468),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.putText(display, "Q = Quit", (550, 468),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Recent predictions
    if prediction_history:
        recent = list(prediction_history)[-5:]
        cv2.putText(display, f"Recent: {' → '.join(recent)}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("PSL Translator v3 - Live Test", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()

print(f"\n{'=' * 60}")
print("  TEST COMPLETE")
print(f"{'=' * 60}")
if prediction_history:
    print(f"   Words detected: {' → '.join(prediction_history)}")
print(f"   Model: {MODEL_PATH}")
print(f"\n   If webcam works well, next test with mobile video.")
print(f"{'=' * 60}")