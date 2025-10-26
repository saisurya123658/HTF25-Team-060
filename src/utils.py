import cv2
import os
import json
from datetime import datetime

def centroid(box):
    x1, y1, x2, y2 = box
    return ((x1+x2)//2, (y1+y2)//2)

def save_violation(frame, violation_type, track_id, output_dir="output"):
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/clips", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = f"{output_dir}/images/{violation_type}_{track_id}_{timestamp}.jpg"
    cv2.imwrite(img_path, frame)
    event = {
        "track_id": track_id,
        "violation_type": violation_type,
        "timestamp": timestamp,
        "image_path": img_path
    }
    events_file = f"{output_dir}/events.json"
    if os.path.exists(events_file):
        with open(events_file, "r") as f:
            events = json.load(f)
    else:
        events = []
    events.append(event)
    with open(events_file, "w") as f:
        json.dump(events, f, indent=4)
    print(f"[INFO] Violation saved: {violation_type} | Track {track_id}")
