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
from tensorflow import keras
import json
import os
import tempfile
from collections import deque
import re
import time

app = Flask(__name__)
CORS(app) # Enable CORS for mobile access

# ============================================================================
#  KNOWLEDGE BASE (Your 55 Valid Sentences)
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
    "I send mine message.",
    "You make project idea.",
    "Boss send project message.",
    "Team give team word.",
    "I talk work office.",
    "We make plan office.",
    "We send word office.",
    "Boss talk word job.",
    "Team talk project word.",
    "Client send idea message.",
    "Boss send message work.",
    "Team write word plan.",
    "Client give boss plan.",
    "Boss make message project.",
    "You send job word.",
    "Client write office project.",
    "Team write report plan.",
    "Client talk message office.",
    "You write office report.",
    "We write job message.",
    "I make idea word.",
    "Boss send job word.",
    "Boss give client message.",
    "Boss make office report.",
    "You write report work.",
    "Team make office message.",
    "Boss talk idea word.",
    "We write idea project.",
    "You write idea project.",
    "I make report job.",
    "Team write job project.",
    "Team send job project.",
    "Boss send work report.",
    "I give client office.",
    "Boss write message idea.",
    "Team write message word.",
    "I send office job.",
    "Boss talk plan project.",
    "Boss give client word.",
    "I write word work.",
    "Boss talk job plan.",
    "Team talk office plan.",
    "Team write project idea.",
    "Boss send message plan.",
    "Team send job office.",
    "Client talk idea word.",
    "Boss talk office project.",
    "I talk word job.",
    "Client write job idea.",
    "Client talk word work.",
    "We send office plan.",
    "I write this report.",
    "Boss write this report.",
    "Team send this idea.",
    "We make this office.",
    "I write this project.",
    "You talk this office.",
    "We make this message.",
    "Boss write this idea.",
    "I make this job.",
    "Client talk this job.",
    "Client read this work.",
    "I read this work.",
    "We talk this job.",
    "Boss send this job.",
    "I read this plan.",
    "We make this report.",
    "Boss make this plan.",
    "Client write this plan.",
    "Boss send this work.",
    "We talk this office.",
    "I write this idea.",
    "Boss make this word.",
    "I write this office.",
    "Boss write this message.",
    "We talk this idea.",
    "Team talk this message.",
    "Team write this idea.",
    "Boss talk this project.",
    "Team meet this boss.",
    "We meet this female.",
    "You call this male.",
    "I meet this female.",
    "We call this team.",
    "Boss call this team.",
    "I call this team.",
    "I call this male.",
    "Client call this team.",
    "Team call this team.",
    "Client meet this team.",
    "Client meet this day.",
    "I meet this time.",
    "Team work this day.",
    "You meet this time.",
    "Client work this day.",
    "I meet this day.",
    "Team work this time.",
    "We work this time.",
    "Boss meet this time.",
    "Boss work this time.",
    "Boss meet this day.",
    "Client work this time.",
    "I work this day.",
    "Team meet this time.",
    "We work this day.",
    "Client meet this time.",
    "Boss work this day.",
    "You meet this day.",
    "Team call female team.",
    "You help male client.",
    "Team help male client.",
    "We help male team.",
    "I help male client.",
    "Client meet male client.",
    "I meet male client.",
    "We meet female team.",
    "I call female client.",
    "I call male client.",
    "Boss help female client.",
    "Male client talk office.",
    "Male client read office.",
    "Male team read word.",
    "Male team make message.",
    "Female client talk now.",
    "Female client work now.",
    "Male client work now.",
    "Male team work now.",
    "Give male this word.",
    "Give boss this message.",
    "Give client this office.",
    "Give team this report.",
    "Give you this message.",
    "Give boss this idea.",
    "Give you this job.",
    "Give you this idea.",
    "Give client this job.",
    "Give boss this work.",
    "Give you this plan.",
    "Talk this message office.",
    "Read this work report.",
    "Write this message plan.",
    "Talk this report job.",
    "Read this report message.",
    "Write this job report.",
    "Write this office plan.",
    "Talk this office project.",
    "Read this message word.",
    "Write this message word.",
    "Write this idea job.",
    "Talk this plan work.",
    "Read this message report.",
    "Read this idea word.",
    "Work this time now.",
    "Meet this day now.",
    "Make this office now.",
    "Make this message now.",
    "Write this plan now.",
    "Talk this report now.",
    "Make this plan now.",
    "Write this word now.",
    "Talk this office now.",
    "Help write this idea.",
    "Help make this plan.",
    "Help send this office.",
    "Help give this office.",
    "Help talk this idea.",
    "Help send this job.",
    "Help read this plan.",
    "This report is mine.",
    "This message is mine.",
    "This job is mine.",
    "This word is mine.",
    "This plan is mine.",
    "This office is mine.",
    "This idea is mine.",
    "Team read mine word.",
    "We read mine job.",
    "Client make mine word.",
    "Boss give mine word.",
    "We make mine project.",
    "You write mine job.",
    "We make mine message.",
    "You read mine office.",
    "We read mine word.",
    "Client write mine report.",
    "I make mine word.",
    "You read mine work.",
    "I make mine job.",
    "We send mine office."
]

# ============================================================================
# CONFIG
# ============================================================================
MODEL_PATH = "psl_model_v3.h5"
CLASS_FILE = "class_names_v3.json"
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.70

# ============================================================================
# LOAD AI ENGINE
# ============================================================================
model = keras.models.load_model(MODEL_PATH)
with open(CLASS_FILE, "r") as f:
    class_names = json.load(f)

mp_holistic = mp.solutions.holistic

# ============================================================================
# BUILD DETECTOR ONCE AT STARTUP (reused across all requests)
# ============================================================================
holistic_detector = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("\u2705 MediaPipe Holistic detector built once at startup.")

# ============================================================================
# WARM UP KERAS MODEL (avoid first-call graph tracing during real requests)
# ============================================================================
_dummy_input = np.zeros((1, SEQUENCE_LENGTH, 144), dtype=np.float32)
_ = model(_dummy_input, training=False)
print("\u2705 Keras model warmed up with dummy forward pass.")

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

# ============================================================================
# VIDEO PROCESSING
# ============================================================================
def process_video(video_path):
    total_start = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return ""

    # Reuse the global holistic detector (no graph rebuild cost)
    global holistic_detector
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prediction_history = []
    frame_count = 0
    mediapipe_time = 0
    prediction_time = 0
    last_results = None
    mp_calls = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # STEP 1+3: Process every 3rd frame, reuse landmarks for skipped frames
        # At 30fps, hand positions barely change across 2-3 consecutive frames
        if frame_count % 3 == 1 or last_results is None:
            mp_start = time.time()
            results = holistic_detector.process(rgb)
            mediapipe_time += time.time() - mp_start
            last_results = results
            mp_calls += 1
        else:
            results = last_results  # Reuse previous frame's landmarks
        
        hands_visible = (results.left_hand_landmarks or results.right_hand_landmarks)
        p, l, r, a = extract_features(results)
        norm = normalize_frame(p, l, r, a)
        
        if norm is not None: frame_buffer.append(norm)
        else: frame_buffer.append(np.zeros(144))
        
        if len(frame_buffer) == SEQUENCE_LENGTH and hands_visible:
            # STEP 2: Predict every 10th frame instead of every 5th
            if frame_count % 10 == 0:
                sequence = np.array(list(frame_buffer))
                inp = np.expand_dims(sequence, axis=0)
                
                pred_start = time.time()
                # HUGE SPEEDUP: Use model(inp) instead of model.predict(inp)
                probs = model(inp, training=False).numpy()[0]
                prediction_time += time.time() - pred_start
                
                idx = np.argmax(probs)
                conf = float(probs[idx])
                pred = class_names[idx]
                
                if pred != "_idle_" and conf >= CONFIDENCE_THRESHOLD:
                    prediction_history.append(pred)

    cap.release()

    print(f"\u23f1\ufe0f [Profiler] Total frames: {frame_count} | MediaPipe calls: {mp_calls}")
    print(f"\u23f1\ufe0f [Profiler] MediaPipe time: {mediapipe_time:.2f}s")
    print(f"\u23f1\ufe0f [Profiler] Prediction time: {prediction_time:.2f}s")
    print(f"\u23f1\ufe0f [Profiler] Total processing: {time.time() - total_start:.2f}s")

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
    request_start = time.time()
    if "video" not in request.files:
        return jsonify({"error": "No video file received"}), 400

    video_file = request.files["video"]
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "received_video.mp4")
    
    save_start = time.time()
    video_file.save(temp_path)
    print(f"\u23f1\ufe0f [Profiler] File save to disk: {time.time() - save_start:.2f}s")

    sentence = process_video(temp_path)

    try:
        os.remove(temp_path)
        os.rmdir(temp_dir)
    except: pass

    print(f"\u23f1\ufe0f [Profiler] Total request time: {time.time() - request_start:.2f}s")

    if sentence:
        return jsonify({"sentence": sentence})
    else:
        return jsonify({"sentence": "", "error": "No signs detected"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)