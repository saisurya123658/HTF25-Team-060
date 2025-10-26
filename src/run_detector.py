import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import easyocr
from utils import centroid
from pathlib import Path
import warnings
import json
import time
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import threading
import os

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Load models ---
print("[INFO] Loading YOLOv8 models...")
model = YOLO("models/yolov8n.pt")       # vehicle/person detection
helmet_model = YOLO("models/helmet.pt")  # helmet detection
print("[INFO] Models loaded successfully.")

# --- Initialize tracker ---
tracker = DeepSort(max_age=30)
print("[INFO] Tracker initialized.")

# --- OCR for license plates ---
reader = easyocr.Reader(['en'])

# --- Video input ---
data_folder = Path("../data")
video_files = list(data_folder.glob("*.mp4"))
if not video_files:
    print("[ERROR] No .mp4 video found in data/")
    exit()
video_path = str(video_files[0])
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERROR] Could not open video: {video_path}")
    exit()

# --- ROI and reference lines ---
line_y = 300  # signal jump line
signal_roi = (500, 50, 550, 150)
lane_lines = [200, 400]
speed_limit = 30  # pixels/frame

# --- Storage ---
prev_centroids = {}
prev_times = {}
violation_folder = Path(r"C:\Users\SURYA TEJA\Desktop\AI_Traffic_Violation_Detector\violations")
violation_folder.mkdir(exist_ok=True)
json_file = violation_folder / "violations.json"
all_violations = []

clip_frames = {}
frame_count = 0
ocr_skip = 5
helmet_skip = 2

# --- Function to save clips asynchronously ---
def save_clip_async(tid, frames, path):
    try:
        clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=10)
        clip.write_videofile(str(path), codec='libx264', audio=False, verbose=False, logger=None)
    except Exception as e:
        print(f"[ERROR] Clip save failed for {tid}: {e}")

print("[INFO] Starting detection loop... Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video.")
        break

    frame_count += 1
    h, w = frame.shape[:2]
    frame_time = time.time()

    # --- YOLO Detection ---
    results = model(frame, imgsz=640)[0]
    detections = []
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        label = int(cls)
        if label in [0, 1, 2, 3, 5, 7]:
            x1, y1, x2, y2 = map(int, box)
            detections.append(([x1, y1, x2, y2], conf, label))

    tracks = tracker.update_tracks(detections, frame=frame)

    # --- Signal detection ---
    x1s, y1s, x2s, y2s = signal_roi
    roi = frame[y1s:y2s, x1s:x2s]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    signal_state = "RED" if mask_red.sum() / (roi.shape[0] * roi.shape[1] * 255) > 0.02 else "GREEN"

    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cX, cY = centroid([x1, y1, x2, y2])

        # Safe crop
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        vehicle_crop = frame[y1c:y2c, x1c:x2c].copy()
        if vehicle_crop.size == 0:
            prev_centroids[tid] = (cX, cY)
            prev_times[tid] = frame_time
            continue

        # --- License Plate (OCR) ---
        if frame_count % ocr_skip == 0:
            lp_texts = reader.readtext(vehicle_crop)
            lp = lp_texts[0][1] if lp_texts else "UNKNOWN"
        else:
            lp = "UNKNOWN"

        violation_type = None
        helmet_violation = False

        # Signal jump
        if tid in prev_centroids:
            _, prevY = prev_centroids[tid]
            if prevY < line_y <= cY and signal_state == "RED":
                violation_type = "signal_violation"

        # Overspeeding
        if tid in prev_centroids and tid in prev_times:
            prevX, prevY = prev_centroids[tid]
            prev_time = prev_times[tid]
            dist = np.sqrt((cX - prevX) ** 2 + (cY - prevY) ** 2)
            dt = frame_time - prev_time
            speed = dist / dt if dt > 0 else 0
            if speed > speed_limit:
                violation_type = "overspeeding"

        # Wrong lane
        if not (lane_lines[0] < cX < lane_lines[1]):
            violation_type = "wrong_lane"

        # Triple riding + helmetless
        if label == 3:
            results_person = [int(cls) for cls in results.boxes.cls if int(cls) == 0]
            if len(results_person) > 2:
                violation_type = "triple_riding"

            if frame_count % helmet_skip == 0:
                head_crop = vehicle_crop[0:int((y2c - y1c) / 2), :]
                helmet_results = helmet_model(head_crop, imgsz=320)[0]
                helmet_detected = any(int(cls) == 0 for cls in helmet_results.boxes.cls)
                if not helmet_detected:
                    helmet_violation = True
                    violation_type = violation_type or "helmet_violation"

        # --- Save violation ---
        if violation_type:
            fname = violation_folder / f"{violation_type}_{frame_time:.0f}_{tid}.jpg"
            cv2.putText(vehicle_crop, f"LP:{lp}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(str(fname), vehicle_crop)

            clip_frames.setdefault(tid, []).append(frame.copy())
            clip_path = None
            if len(clip_frames[tid]) > 10:
                clip_path = violation_folder / f"clip_{violation_type}_{frame_time:.0f}_{tid}.mp4"
                threading.Thread(target=save_clip_async, args=(tid, clip_frames[tid], clip_path), daemon=True).start()
                clip_frames[tid] = []

            all_violations.append({
                "violation_type": violation_type,
                "track_id": tid,
                "license_plate": lp,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": str(fname.resolve()),
                "clip_path": str(clip_path.resolve()) if clip_path else None,
                "helmet_violation": helmet_violation
            })

        prev_centroids[tid] = (cX, cY)
        prev_times[tid] = frame_time

        # Draw boxes
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("AI Traffic Violation Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# --- Save JSON ---
with open(json_file, "w") as f:
    json.dump(all_violations, f, indent=4)

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Detection finished. Violations saved at {json_file}")
