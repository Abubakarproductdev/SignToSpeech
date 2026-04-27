# ============================================================================
# RAILWAY SERVER.PY - SMART MATCH EDITION
# PSL Translator - Final Year Project
#
# Logic: Video -> Model Prediction -> Keyword Matching -> Correct Sentence
# ============================================================================

#main approach is to predictic and then match from the selected sentetnces.



from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import json
import os
import tempfile
from collections import deque
import re
import threading
import time

app = Flask(__name__)
CORS(app) # Enable CORS for mobile access

# ============================================================================
# 🧠 KNOWLEDGE BASE (Your 55 Valid Sentences)
# ============================================================================
VALID_SENTENCES = [
    "Boss send project report.",
    "I write this plan.",
    "Team make office work.",
    "Give client this message.",
    "Read this job report.",
    "I send project word.",
    "We write office plan.",
    "Client read this message.",
    "Boss give team job.",
    "Write this work report.",
    "Send client project idea.",
    "Team read this plan.",
    "I make job report.",
    "Give boss this word.",
    "We send office message.",
    "We meet this day.",
    "I call this client.",
    "Talk this project word.",
    "Boss meet female client.",
    "Male team talk work.",
    "I call office team.",
    "We meet this time.",
    "You talk this idea.",
    "Boss call male client.",
    "Team meet this day.",
    "I talk project plan.",
    "We call this boss.",
    "Female team meet now.",
    "Talk this job now.",
    "You meet this client.",
    "I give you idea.",
    "Help make this project.",
    "You send mine report.",
    "We make this plan.",
    "Give male this idea.",
    "You help female team.",
    "I make this word.",
    "We give client idea.",
    "You send this message.",
    "Help write this report.",
    "I give team work.",
    "You make this job.",
    "Work this day now.",
    "Meet this time now.",
    "I work this time.",
    "We send report day.",
    "Make project plan time.",
    "Write this word day.",
    "You work this day.",
    "Team meet work time.",
    "This work is mine.",
    "I give mine idea.",
    "You read mine report.",
    "This project is mine.",
    "I send mine message."
]

# ============================================================================
# CONFIG
# ============================================================================
MODEL_PATH = "psl_model_v3.h5"
TFLITE_MODEL_PATH = "psl_model_v3.tflite"
CLASS_FILE = "class_names_v3.json"
DATASET_SAMPLES_PATH = "preprocessed_dataset_v3.npz"
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.70
HIGH_FPS_THRESHOLD = 24.0
HIGH_FPS_PREDICTION_STRIDE = 2
LOW_FPS_PREDICTION_STRIDE = 1
TRAILING_NO_HAND_SECONDS = 0.75
TFLITE_CONFIDENCE_DELTA_TOLERANCE = 0.05
TFLITE_VALIDATION_SAMPLE_LIMIT = 3
ZERO_FRAME = np.zeros(144, dtype=np.float32)
WARMUP_SEQUENCE = np.zeros((1, SEQUENCE_LENGTH, 144), dtype=np.float32)

# ============================================================================
# LOAD AI ENGINE
# ============================================================================
with open(CLASS_FILE, "r") as f:
    class_names = json.load(f)

mp_holistic = mp.solutions.holistic


class PredictionEngine:
    def __init__(self, keras_model_path, tflite_model_path, validation_dataset_path):
        self.keras_model = keras.models.load_model(keras_model_path)
        self.keras_lock = threading.Lock()
        self.keras_model(WARMUP_SEQUENCE, training=False)
        self.tflite_interpreter = None
        self.tflite_input_details = None
        self.tflite_output_details = None
        self.tflite_lock = threading.Lock()
        self.engine_name = "keras"

        if os.path.exists(tflite_model_path):
            try:
                self._load_tflite(tflite_model_path)
                if self._validate_tflite(validation_dataset_path):
                    self.engine_name = "tflite"
                else:
                    self._disable_tflite("validation mismatch")
            except Exception as exc:
                self._disable_tflite(str(exc))

        print(f"Inference engine: {self.engine_name}")

    def _load_tflite(self, tflite_model_path):
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path, num_threads=max(1, os.cpu_count() or 1))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        interpreter.set_tensor(input_details["index"], WARMUP_SEQUENCE.astype(input_details["dtype"], copy=False))
        interpreter.invoke()

        self.tflite_interpreter = interpreter
        self.tflite_input_details = input_details
        self.tflite_output_details = output_details

    def _disable_tflite(self, reason):
        print(f"TFLite disabled: {reason}")
        self.tflite_interpreter = None
        self.tflite_input_details = None
        self.tflite_output_details = None

    def _load_validation_sequences(self, validation_dataset_path):
        sequences = [WARMUP_SEQUENCE[0]]

        if not os.path.exists(validation_dataset_path):
            return sequences

        try:
            data = np.load(validation_dataset_path)
            samples = data["X"][:TFLITE_VALIDATION_SAMPLE_LIMIT]
            for sample in samples:
                sequences.append(sample.astype(np.float32, copy=False))
        except Exception as exc:
            print(f"TFLite validation samples unavailable: {exc}")

        return sequences

    def _validate_tflite(self, validation_dataset_path):
        if self.tflite_interpreter is None:
            return False

        for sequence in self._load_validation_sequences(validation_dataset_path):
            keras_probs = self._predict_with_keras(sequence)
            tflite_probs = self._predict_with_tflite(sequence)

            keras_idx = int(np.argmax(keras_probs))
            tflite_idx = int(np.argmax(tflite_probs))
            keras_conf = float(keras_probs[keras_idx])
            tflite_conf = float(tflite_probs[tflite_idx])

            if keras_idx != tflite_idx:
                print(f"TFLite validation failed: argmax mismatch {keras_idx} vs {tflite_idx}")
                return False

            if abs(keras_conf - tflite_conf) > TFLITE_CONFIDENCE_DELTA_TOLERANCE:
                print(
                    "TFLite validation failed: confidence drift "
                    f"{keras_conf:.4f} vs {tflite_conf:.4f}"
                )
                return False

        return True

    def _predict_with_keras(self, sequence):
        batch = np.expand_dims(sequence.astype(np.float32, copy=False), axis=0)
        with self.keras_lock:
            return self.keras_model(batch, training=False).numpy()[0]

    def _predict_with_tflite(self, sequence):
        batch = np.expand_dims(sequence.astype(np.float32, copy=False), axis=0)
        with self.tflite_lock:
            self.tflite_interpreter.set_tensor(
                self.tflite_input_details["index"],
                batch.astype(self.tflite_input_details["dtype"], copy=False),
            )
            self.tflite_interpreter.invoke()
            return self.tflite_interpreter.get_tensor(self.tflite_output_details["index"])[0]

    def predict(self, sequence):
        if self.engine_name == "tflite" and self.tflite_interpreter is not None:
            return self._predict_with_tflite(sequence)
        return self._predict_with_keras(sequence)


prediction_engine = PredictionEngine(MODEL_PATH, TFLITE_MODEL_PATH, DATASET_SAMPLES_PATH)

# ============================================================================
# 🧠 SMART MATCHING ALGORITHM
# ============================================================================
def get_best_sentence_match(raw_predicted_words):
    """
    Takes a list of raw words (e.g. ['Boss', 'send', 'word'])
    Returns the closest sentence from VALID_SENTENCES.
    """
    if not raw_predicted_words:
        return ""

    # 1. Clean up predictions (lowercase, remove duplicates)
    predicted_set = set([w.lower().strip() for w in raw_predicted_words])
    
    best_score = 0
    best_sentence = ""

    # 2. Compare against every valid sentence
    for sentence in VALID_SENTENCES:
        # Clean sentence (remove dots, lowercase)
        clean_target = re.sub(r'[^\w\s]', '', sentence).lower()
        target_words = set(clean_target.split())
        
        # Calculate Overlap: How many predicted words exist in this sentence?
        # We use intersection logic
        overlap_count = len(predicted_set.intersection(target_words))
        
        # 3. Keep the best match
        if overlap_count > best_score:
            best_score = overlap_count
            best_sentence = sentence
            
    # 4. Fallback Logic
    # If we found a match with at least 1 word overlap, return it.
    if best_score > 0:
        print(f"✅ Smart Match: Raw='{predicted_set}' -> Matched='{best_sentence}' (Score: {best_score})")
        return best_sentence
    else:
        # If score is 0 (total gibberish), just return the raw words
        raw_sentence = " ".join(raw_predicted_words)
        print(f"⚠️ No Match Found. Returning raw: {raw_sentence}")
        return raw_sentence

# ============================================================================
# FEATURE EXTRACTION
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

    lh = (np.array([[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark]).flatten() 
          if results.left_hand_landmarks else np.zeros(63))
    rh = (np.array([[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark]).flatten() 
          if results.right_hand_landmarks else np.zeros(63))
    return upper_body, lh, rh, anchors

def normalize_frame(pose, lh, rh, anchors):
    l_sh, r_sh = anchors[0], anchors[1]
    if np.sum(l_sh) == 0 or np.sum(r_sh) == 0: return None
    center = (l_sh + r_sh) / 2
    mid_shoulder = (l_sh + r_sh) / 2
    l_hip, r_hip = anchors[2], anchors[3]
    if np.sum(l_hip) != 0 and np.sum(r_hip) != 0:
        mid_hip = (l_hip + r_hip) / 2
        scale = np.linalg.norm(mid_shoulder - mid_hip)
    else:
        scale = np.linalg.norm(l_sh - r_sh) * 1.5
    if scale < 0.1: scale = 1
    def norm(data):
        if len(data) == 0: return data
        reshaped = data.reshape(-1, 3)
        mask = np.any(reshaped != 0, axis=1)
        reshaped[mask] = (reshaped[mask] - center) / scale
        return reshaped.flatten()
    return np.concatenate([norm(pose), norm(lh), norm(rh)])


def get_prediction_stride(video_fps):
    if video_fps and video_fps >= HIGH_FPS_THRESHOLD:
        return HIGH_FPS_PREDICTION_STRIDE
    return LOW_FPS_PREDICTION_STRIDE


def get_trailing_no_hand_break_frames(video_fps):
    fps = video_fps if video_fps and video_fps > 0 else HIGH_FPS_THRESHOLD
    return max(8, int(round(fps * TRAILING_NO_HAND_SECONDS)))

# ============================================================================
# VIDEO PROCESSING
# ============================================================================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return ""

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    prediction_stride = get_prediction_stride(video_fps)
    trailing_no_hand_break_frames = get_trailing_no_hand_break_frames(video_fps)

    holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prediction_history = []
    frame_index = 0
    trailing_no_hand_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        
        hands_visible = (results.left_hand_landmarks or results.right_hand_landmarks)
        p, l, r, a = extract_features(results)
        norm = normalize_frame(p, l, r, a)
        
        if norm is not None: frame_buffer.append(norm.astype(np.float32, copy=False))
        else: frame_buffer.append(ZERO_FRAME)

        if hands_visible:
            trailing_no_hand_frames = 0
        elif prediction_history:
            trailing_no_hand_frames += 1
            if trailing_no_hand_frames >= trailing_no_hand_break_frames:
                break
        
        if (
            len(frame_buffer) == SEQUENCE_LENGTH
            and hands_visible
            and frame_index % prediction_stride == 0
        ):
            sequence = np.stack(frame_buffer, axis=0)
            probs = prediction_engine.predict(sequence)
            idx = np.argmax(probs)
            conf = float(probs[idx])
            pred = class_names[idx]
            
            if pred != "_idle_" and conf >= CONFIDENCE_THRESHOLD:
                prediction_history.append(pred)

        frame_index += 1

    cap.release()
    holistic.close()

    if not prediction_history: return ""

    # 1. Deduplicate raw predictions
    unique_words = []
    last_word = ""
    for pred in prediction_history:
        if pred != last_word:
            unique_words.append(pred)
            last_word = pred

    # 2. RUN INTELLIGENT MATCHING
    final_sentence = get_best_sentence_match(unique_words)
    
    return final_sentence

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.route("/predict_sentence", methods=["POST"])
def predict_sentence():
    if "video" not in request.files:
        return jsonify({"error": "No video file received"}), 400

    video_file = request.files["video"]
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "received_video.mp4")
    video_file.save(temp_path)

    start_time = time.perf_counter()
    sentence = process_video(temp_path)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 1)

    try:
        os.remove(temp_path)
        os.rmdir(temp_dir)
    except: pass

    if sentence:
        return jsonify({"sentence": sentence, "latency_ms": duration_ms, "engine": prediction_engine.engine_name})
    else:
        return jsonify({
            "sentence": "",
            "error": "No signs detected",
            "latency_ms": duration_ms,
            "engine": prediction_engine.engine_name,
        })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine": prediction_engine.engine_name})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
