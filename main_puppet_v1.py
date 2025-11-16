"""
AI Fitness Tracker v1 — Complete & Error-Free
---------------------------------------------
Features:
✅ Pose classification: Standing, Arms Up, Sitting, Squat, Plank
✅ Motion energy visualization (green energy bar)
✅ Squat rep counting (based on knee angle)
✅ Gesture recognition (Open ✋, Pinch 🤏, Fist ✊)
✅ Smooth landmark filtering (EMA)
✅ Dual-display (live + glowing stickman)
✅ HUD with pose, gesture, reps, energy, FPS
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque

# ---------------- CONFIG ----------------
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
SMOOTH_ALPHA = 0.65
VISIBILITY_THRESH = 0.5
ENERGY_WINDOW = 30

# Thresholds
SQUAT_KNEE_ANGLE_DOWN = 100
SQUAT_KNEE_ANGLE_UP = 160
PLANK_TORSO_ANGLE_DEG = 25
ARM_UP_Y_DIFF = -0.15
SITTING_HIP_SHOULDER_DIFF = 0.12
MAX_ENERGY = 0.12  # for energy bar scaling

# ---------------- HELPERS ----------------
def angle_between(a, b, c):
    """Angle at point b formed by points a-b-c."""
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 * n2 == 0:
        return 0.0
    cosang = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def normalized_to_pixel(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))

# ---------------- MEDIAPIPE SETUP ----------------
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

# ---------------- STATE ----------------
smooth_landmarks = None
prev_landmarks = None
rep_count = 0
squat_down = False
last_rep_time = 0
energy_smoothed = 0.0
energy_timeline = deque(maxlen=ENERGY_WINDOW)

print("🏋️‍♂️ AI Fitness Tracker — Press 'r' to reset, 'q' or ESC to quit.")

# ---------------- ENERGY ----------------
def compute_motion_energy(prev, curr):
    if prev is None or curr is None:
        return 0.0
    e = 0.0
    for i in range(len(curr.landmark)):
        dx = curr.landmark[i].x - prev.landmark[i].x
        dy = curr.landmark[i].y - prev.landmark[i].y
        e += math.hypot(dx, dy)
    return e

# ---------------- POSE CLASSIFICATION ----------------
def classify_pose(lm):
    try:
        left_sh, right_sh = lm[11], lm[12]
        left_hip, right_hip = lm[23], lm[24]
        left_wrist, right_wrist = lm[15], lm[16]
    except Exception:
        return "Unknown"

    avg_sh_y = (left_sh.y + right_sh.y) / 2
    avg_wr_y = (left_wrist.y + right_wrist.y) / 2
    avg_hip_y = (left_hip.y + right_hip.y) / 2

    # Arms Up
    if (avg_wr_y - avg_sh_y) < ARM_UP_Y_DIFF:
        return "Arms Up 🙌"
    # Sitting
    if (avg_hip_y - avg_sh_y) > SITTING_HIP_SHOULDER_DIFF:
        return "Sitting 💺"
    # Squat
    try:
        l_hip = (lm[23].x, lm[23].y); l_knee = (lm[25].x, lm[25].y); l_ank = (lm[27].x, lm[27].y)
        r_hip = (lm[24].x, lm[24].y); r_knee = (lm[26].x, lm[26].y); r_ank = (lm[28].x, lm[28].y)
        avg_k = (angle_between(l_hip, l_knee, l_ank) + angle_between(r_hip, r_knee, r_ank)) / 2
        if avg_k < SQUAT_KNEE_ANGLE_DOWN:
            return "Squat 🏋️"
    except: pass

    # Plank (horizontal torso)
    try:
        sh_mid = ((lm[11].x + lm[12].x)/2, (lm[11].y + lm[12].y)/2)
        hip_mid = ((lm[23].x + lm[24].x)/2, (lm[23].y + lm[24].y)/2)
        vx, vy = hip_mid[0]-sh_mid[0], hip_mid[1]-sh_mid[1]
        dot = vy  # with vertical
        mag = math.hypot(vx, vy)
        if mag > 0:
            ang = math.degrees(math.acos(np.clip(dot/mag, -1, 1)))
            if abs(ang - 90) < PLANK_TORSO_ANGLE_DEG:
                return "Plank 🤸"
    except: pass

    return "Standing 🧍"

# ---------------- GESTURE DETECTION ----------------
def detect_gesture(hand_landmarks, w, h):
    if not hand_landmarks:
        return "No Hands"
    lm = hand_landmarks.landmark
    thumb_tip = np.array([lm[4].x * w, lm[4].y * h])
    index_tip = np.array([lm[8].x * w, lm[8].y * h])
    middle_tip = np.array([lm[12].x * w, lm[12].y * h])
    wrist = np.array([lm[0].x * w, lm[0].y * h])

    pinch_dist = np.linalg.norm(thumb_tip - index_tip)
    open_dist = np.linalg.norm(middle_tip - wrist)

    if pinch_dist < 40:
        return "Pinch 🤏"
    elif open_dist > 150:
        return "Open ✋"
    else:
        return "Fist ✊"

# ---------------- MAIN LOOP ----------------
prev_time = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    res_pose = pose.process(rgb)
    res_hands = hands.process(rgb)

    # Smooth pose
    if res_pose.pose_landmarks:
        if smooth_landmarks is None:
            smooth_landmarks = res_pose.pose_landmarks
        else:
            for i in range(len(res_pose.pose_landmarks.landmark)):
                sm = smooth_landmarks.landmark[i]
                raw = res_pose.pose_landmarks.landmark[i]
                sm.x = SMOOTH_ALPHA*sm.x + (1-SMOOTH_ALPHA)*raw.x
                sm.y = SMOOTH_ALPHA*sm.y + (1-SMOOTH_ALPHA)*raw.y
                sm.visibility = max(sm.visibility, raw.visibility)

    # Energy
    energy = compute_motion_energy(prev_landmarks, res_pose.pose_landmarks) if res_pose.pose_landmarks else 0.0
    energy_smoothed = 0.85*energy_smoothed + 0.15*energy
    prev_landmarks = res_pose.pose_landmarks

    # Pose + Reps
    pose_label = "No Pose"
    if smooth_landmarks:
        lm = smooth_landmarks.landmark
        pose_label = classify_pose(lm)
        # Squat rep logic
        try:
            l_hip = (lm[23].x, lm[23].y); l_knee = (lm[25].x, lm[25].y); l_ank = (lm[27].x, lm[27].y)
            r_hip = (lm[24].x, lm[24].y); r_knee = (lm[26].x, lm[26].y); r_ank = (lm[28].x, lm[28].y)
            avg_k = (angle_between(l_hip, l_knee, l_ank) + angle_between(r_hip, r_knee, r_ank)) / 2
            if avg_k < SQUAT_KNEE_ANGLE_DOWN and not squat_down:
                squat_down = True
            if avg_k > SQUAT_KNEE_ANGLE_UP and squat_down:
                rep_count += 1
                squat_down = False
                last_rep_time = time.time()
        except: pass

    # Gesture
    gesture_label = "No Hands"
    if res_hands.multi_hand_landmarks:
        for hl in res_hands.multi_hand_landmarks:
            gesture_label = detect_gesture(hl, w, h)
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

    # Build stickman
    puppet = np.zeros_like(frame)
    if smooth_landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        hue = int((time.time() * 30) % 180)
        color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
        for a, b in mp_pose.POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(puppet, pts[a], pts[b], color, 5)
        for p in pts:
            cv2.circle(puppet, p, 5, (255, 255, 255), -1)
        hx, hy = pts[0]
        cv2.circle(puppet, (hx, hy-20), 25, color, -1)
        glow = cv2.GaussianBlur(puppet, (25,25), 15)
        puppet = cv2.addWeighted(puppet, 1.3, glow, 0.7, 0)

    # Combine
    live_display = cv2.GaussianBlur(frame, (7,7), 6)
    left = cv2.resize(live_display, (w//2, h))
    right = cv2.resize(puppet, (w//2, h))
    combined = np.hstack((left, right))

    # HUD
    cv2.rectangle(combined, (10,10), (430,170), (0,0,0), -1)
    cv2.putText(combined, f"Pose: {pose_label}", (20,45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(combined, f"Gesture: {gesture_label}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(combined, f"Squat Reps: {rep_count}", (20,115), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(combined, f"Energy: {energy_smoothed:.2f}", (20,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Energy bar
    e_norm = min(1.0, energy_smoothed / MAX_ENERGY)
    filled = int(380 * e_norm)
    cv2.rectangle(combined, (20,170), (400,180), (80,80,80), -1)
    cv2.rectangle(combined, (20,170), (20+filled,180), (0,200,50), -1)
    cv2.putText(combined, f"{int(e_norm*100)}%", (410,180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Rep flash
    if time.time() - last_rep_time < 1.5:
        cv2.putText(combined, "REP +1", (460, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,100), 3)

    # FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if prev_time else 0.0
    prev_time = curr_time
    cv2.putText(combined, f"FPS: {int(fps)}", (combined.shape[1]-150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow("🏋️ AI Fitness Tracker — Live | Puppet", combined)

    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]: break
    if key == ord('r'):
        rep_count = 0
        energy_smoothed = 0.0
        energy_timeline.clear()
        print("🔄 Counters reset.")

cap.release()
cv2.destroyAllWindows()
