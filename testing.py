# ============================================================================
# COMPARE: What does webcam see vs what does mobile see?
#
# Records 3 seconds from WEBCAM, then asks you to send from MOBILE.
# Compares the MediaPipe coordinates side by side.
#
# This will tell us EXACTLY why predictions differ.
# ============================================================================

import cv2
import numpy as np
import mediapipe as mp
import os

mp_holistic = mp.solutions.holistic

def extract_coords(results):
    """Get raw shoulder + hand positions (NOT normalized)."""
    info = {}
    
    if results.pose_landmarks:
        res = results.pose_landmarks.landmark
        info["L_shoulder"] = (round(res[11].x, 3), round(res[11].y, 3))
        info["R_shoulder"] = (round(res[12].x, 3), round(res[12].y, 3))
        info["L_wrist"]    = (round(res[15].x, 3), round(res[15].y, 3))
        info["R_wrist"]    = (round(res[16].x, 3), round(res[16].y, 3))
        
        # Hip positions
        info["L_hip"] = (round(res[23].x, 3), round(res[23].y, 3))
        info["R_hip"] = (round(res[24].x, 3), round(res[24].y, 3))
        
        # Nose (head position)
        info["Nose"] = (round(res[0].x, 3), round(res[0].y, 3))
    else:
        info["L_shoulder"] = None
        info["R_shoulder"] = None
    
    info["LH_detected"] = results.left_hand_landmarks is not None
    info["RH_detected"] = results.right_hand_landmarks is not None
    
    if results.left_hand_landmarks:
        lh = results.left_hand_landmarks.landmark
        info["LH_center"] = (round(lh[9].x, 3), round(lh[9].y, 3))
    
    if results.right_hand_landmarks:
        rh = results.right_hand_landmarks.landmark
        info["RH_center"] = (round(rh[9].x, 3), round(rh[9].y, 3))
    
    return info


def analyze_video(source, label):
    """Analyze a video source and return average positions."""
    
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print(f"   ❌ Cannot open: {source}")
        return None
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n{'━' * 60}")
    print(f"   {label}")
    print(f"{'━' * 60}")
    print(f"   Resolution: {w}x{h} @ {fps:.0f} FPS")
    
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    all_coords = []
    frame_count = 0
    lh_count = 0
    rh_count = 0
    pose_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        
        coords = extract_coords(results)
        all_coords.append(coords)
        
        if coords["L_shoulder"] is not None:
            pose_count += 1
        if coords["LH_detected"]:
            lh_count += 1
        if coords["RH_detected"]:
            rh_count += 1
        
        frame_count += 1
    
    cap.release()
    holistic.close()
    
    print(f"   Frames: {frame_count}")
    print(f"   Pose: {pose_count}/{frame_count} ({pose_count/max(frame_count,1)*100:.0f}%)")
    print(f"   Left hand: {lh_count}/{frame_count} ({lh_count/max(frame_count,1)*100:.0f}%)")
    print(f"   Right hand: {rh_count}/{frame_count} ({rh_count/max(frame_count,1)*100:.0f}%)")
    
    # Average positions
    l_sh_y = []
    r_sh_y = []
    l_sh_x = []
    r_sh_x = []
    nose_y = []
    hip_y = []
    
    for c in all_coords:
        if c["L_shoulder"]:
            l_sh_x.append(c["L_shoulder"][0])
            l_sh_y.append(c["L_shoulder"][1])
            r_sh_x.append(c["R_shoulder"][0])
            r_sh_y.append(c["R_shoulder"][1])
        if c.get("Nose"):
            nose_y.append(c["Nose"][1])
        if c.get("L_hip"):
            hip_y.append(c["L_hip"][1])
    
    if l_sh_y:
        avg_l_sh = (np.mean(l_sh_x), np.mean(l_sh_y))
        avg_r_sh = (np.mean(r_sh_x), np.mean(r_sh_y))
        avg_nose_y = np.mean(nose_y) if nose_y else 0
        avg_hip_y = np.mean(hip_y) if hip_y else 0
        
        shoulder_mid_y = (avg_l_sh[1] + avg_r_sh[1]) / 2
        shoulder_mid_x = (avg_l_sh[0] + avg_r_sh[0]) / 2
        body_span = avg_hip_y - avg_nose_y if avg_hip_y and avg_nose_y else 0
        
        print(f"\n   BODY POSITION (0.0=top/left, 1.0=bottom/right):")
        print(f"   ┌─────────────────────────────────────────────┐")
        print(f"   │  Nose Y:          {avg_nose_y:.3f}                     │")
        print(f"   │  L.Shoulder:      ({avg_l_sh[0]:.3f}, {avg_l_sh[1]:.3f})          │")
        print(f"   │  R.Shoulder:      ({avg_r_sh[0]:.3f}, {avg_r_sh[1]:.3f})          │")
        print(f"   │  Shoulder center: ({shoulder_mid_x:.3f}, {shoulder_mid_y:.3f})          │")
        print(f"   │  Hip Y:           {avg_hip_y:.3f}                     │")
        print(f"   │  Body span:       {body_span:.3f}                     │")
        print(f"   └─────────────────────────────────────────────┘")
        
        # Verdict
        print(f"\n   VERDICT:")
        if shoulder_mid_y > 0.75:
            print(f"   ❌ Body is TOO LOW (shoulders at {shoulder_mid_y:.0%} of frame)")
            print(f"      Should be around 40-60%")
        elif shoulder_mid_y < 0.25:
            print(f"   ❌ Body is TOO HIGH (shoulders at {shoulder_mid_y:.0%} of frame)")
        else:
            print(f"   ✅ Body position OK (shoulders at {shoulder_mid_y:.0%} of frame)")
        
        if body_span < 0.15:
            print(f"   ❌ Body too SMALL in frame (span: {body_span:.3f})")
            print(f"      Move CLOSER to camera")
        elif body_span > 0.7:
            print(f"   ⚠️ Body very LARGE in frame (span: {body_span:.3f})")
        else:
            print(f"   ✅ Body size OK (span: {body_span:.3f})")
        
        return {
            "shoulder_y": shoulder_mid_y,
            "shoulder_x": shoulder_mid_x,
            "body_span": body_span,
            "nose_y": avg_nose_y,
            "hip_y": avg_hip_y,
            "lh_pct": lh_count / max(frame_count, 1),
            "rh_pct": rh_count / max(frame_count, 1),
        }
    
    return None


# ============================================================================
# MAIN
# ============================================================================
print("=" * 60)
print("  WEBCAM vs MOBILE COMPARISON")
print("=" * 60)

# --- STEP 1: Analyze a WEBCAM training video ---
print("\n\n📹 ANALYZING WEBCAM TRAINING VIDEO...")

# Find a webcam video from dataset
dataset_path = os.path.join(os.getcwd(), "dataset")
webcam_video = None
mobile_video = None

for cls in sorted(os.listdir(dataset_path)):
    cls_path = os.path.join(dataset_path, cls)
    if not os.path.isdir(cls_path):
        continue
    for f in os.listdir(cls_path):
        if "webcam" in f.lower() and f.endswith(".mp4"):
            if webcam_video is None:
                webcam_video = os.path.join(cls_path, f)
                webcam_class = cls
        if "1080p" in f.lower() and f.endswith(".mp4"):
            if mobile_video is None:
                mobile_video = os.path.join(cls_path, f)
                mobile_class = cls

if webcam_video:
    print(f"   Found: {webcam_video}")
    webcam_stats = analyze_video(webcam_video, f"WEBCAM TRAINING VIDEO ({webcam_class})")
else:
    print("   ❌ No webcam video found in dataset")
    webcam_stats = None

# --- STEP 2: Analyze a MOBILE training video ---
if mobile_video:
    print(f"\n\n📱 ANALYZING MOBILE TRAINING VIDEO...")
    print(f"   Found: {mobile_video}")
    mobile_train_stats = analyze_video(mobile_video, f"MOBILE TRAINING VIDEO ({mobile_class})")
else:
    print("\n   ❌ No mobile (1080p) video found in dataset")
    mobile_train_stats = None

# --- STEP 3: Analyze the debug frame from server ---
debug_frame = os.path.join(os.getcwd(), "debug_first_frame.jpg")
if os.path.exists(debug_frame):
    print(f"\n\n📱 ANALYZING MOBILE APP FRAME (debug_first_frame.jpg)...")
    
    # Read just the single frame
    frame = cv2.imread(debug_frame)
    h, w = frame.shape[:2]
    print(f"   Resolution: {w}x{h}")
    
    holistic = mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)
    coords = extract_coords(results)
    holistic.close()
    
    if coords["L_shoulder"]:
        print(f"\n   MOBILE APP FRAME POSITIONS:")
        print(f"   L.Shoulder: {coords['L_shoulder']}")
        print(f"   R.Shoulder: {coords['R_shoulder']}")
        print(f"   Nose:       {coords.get('Nose', 'N/A')}")
        print(f"   L.Hip:      {coords.get('L_hip', 'N/A')}")
        print(f"   LH: {'✅' if coords['LH_detected'] else '❌'}")
        print(f"   RH: {'✅' if coords['RH_detected'] else '❌'}")

# --- STEP 4: COMPARISON ---
print(f"\n\n{'━' * 60}")
print(f"   FINAL COMPARISON")
print(f"{'━' * 60}")

print(f"\n   {'METRIC':<25} {'WEBCAM':>12} {'MOB TRAIN':>12} {'MOB APP':>12}")
print(f"   {'─'*25} {'─'*12} {'─'*12} {'─'*12}")

if webcam_stats:
    print(f"   {'Shoulder Y':<25} {webcam_stats['shoulder_y']:>12.3f}", end="")
else:
    print(f"   {'Shoulder Y':<25} {'N/A':>12}", end="")

if mobile_train_stats:
    print(f" {mobile_train_stats['shoulder_y']:>12.3f}", end="")
else:
    print(f" {'N/A':>12}", end="")

print(f" {'see above':>12}")

if webcam_stats and mobile_train_stats:
    print(f"   {'Body span':<25} {webcam_stats['body_span']:>12.3f} {mobile_train_stats['body_span']:>12.3f}")
    print(f"   {'Left hand %':<25} {webcam_stats['lh_pct']:>11.0%} {mobile_train_stats['lh_pct']:>11.0%}")
    print(f"   {'Right hand %':<25} {webcam_stats['rh_pct']:>11.0%} {mobile_train_stats['rh_pct']:>11.0%}")

    diff = abs(webcam_stats['shoulder_y'] - mobile_train_stats['shoulder_y'])
    print(f"\n   Shoulder Y difference: {diff:.3f}")
    
    if diff > 0.15:
        print(f"   ❌ LARGE DIFFERENCE — this is likely the problem!")
        print(f"      Webcam shoulders at {webcam_stats['shoulder_y']:.0%} of frame")
        print(f"      Mobile shoulders at {mobile_train_stats['shoulder_y']:.0%} of frame")
        print(f"      The model learned different body positions for each source")
    else:
        print(f"   ✅ Similar body positions in training data")

print(f"\n{'━' * 60}")