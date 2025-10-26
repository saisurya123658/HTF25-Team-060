
# import os, json
# from datetime import datetime

# # Path to violations folder
# violations_dir = r"C:\Users\SURYA TEJA\Desktop\AI_Traffic_Violation_Detector\violations"
# os.makedirs(violations_dir, exist_ok=True)

# # Supported extensions
# valid_exts = [".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"]

# # List all files (including those without extension)
# files = [f for f in os.listdir(violations_dir) if not f.startswith(".")]

# # Helper function to detect violation type from filename
# def detect_violation_type(filename):
#     name = filename.lower()
#     if "helmet" in name:
#         return "Helmetless Riding"
#     elif "signal" in name or "redlight" in name:
#         return "Signal Jumping"
#     elif "lane" in name:
#         return "Wrong Lane Driving"
#     elif "speed" in name:
#         return "Overspeeding"
#     elif "triple" in name or "three" in name:
#         return "Triple Riding"
#     else:
#         return "Unknown"

# # Build JSON data
# data = []
# for idx, f in enumerate(files, start=1):
#     violation_type = detect_violation_type(f)
#     full_path = os.path.join(violations_dir, f)

#     # Skip folders
#     if os.path.isdir(full_path):
#         continue

#     # Debug log
#     print(f"Processing {f} → Detected type: {violation_type}")

#    # OCR detection code (example)
# lp_texts = reader.readtext(vehicle_crop)   # your EasyOCR result
# lp = lp_texts[0][1] if lp_texts else "UNKNOWN"

# data.append({
#     "track_id": idx,
#     "violation_type": violation_type,
#     "timestamp": str(datetime.now()),
#     "image_path": full_path,
#     "license_plate": lp   # <-- Add this line
# })

# # Save JSON
# json_path = os.path.join(violations_dir, "violations.json")
# with open(json_path, "w") as jf:
#     json.dump(data, jf, indent=4)

# print(f"\n✅ Generated {len(data)} violations in violations.json")
# import os, json
# from datetime import datetime

# violations_dir = r"C:\Users\SURYA TEJA\Desktop\AI_Traffic_Violation_Detector\violations"
# os.makedirs(violations_dir, exist_ok=True)

# valid_exts = [".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"]
# files = [f for f in os.listdir(violations_dir) if os.path.splitext(f)[1].lower() in valid_exts]

# def detect_violation_type(filename):
#     name = filename.lower()
#     if "helmet" in name:
#         return "Helmetless Riding"
#     elif "overspeed" in name or "speed" in name:
#         return "Overspeeding"
#     elif "triple" in name or "three" in name:
#         return "Triple Riding"
#     elif "signal" in name or "redlight" in name or "jump" in name:
#         return "Signal Jumping"
#     elif "wrong_lane" in name or "lane" in name:
#         return "Wrong Lane Driving"
#     else:
#         return "Unknown"

# data = []
# for idx, f in enumerate(files, start=1):
#     violation_type = detect_violation_type(f)
#     data.append({
#         "track_id": idx,
#         "violation_type": violation_type,
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "image_path": os.path.join(violations_dir, f)
#     })

# json_path = os.path.join(violations_dir, "violations.json")
# with open(json_path, "w") as jf:
#     json.dump(data, jf, indent=4)

# print(f"✅ Generated {len(data)} violations in violations.json")
# import os
# import json
# from datetime import datetime

# # ✅ Path to violations folder
# violations_dir = os.path.join(os.getcwd(), "violations")
# os.makedirs(violations_dir, exist_ok=True)

# valid_exts = [".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"]

# def detect_violation_type(filename):
#     name = filename.lower()
#     if "helmet" in name:
#         return "Helmetless Riding"
#     elif "overspeed" in name or "speed" in name:
#         return "Overspeeding"
#     elif "triple" in name or "three" in name:
#         return "Triple Riding"
#     elif "signal" in name or "redlight" in name or "jump" in name:
#         return "Signal Jumping"
#     elif "lane" in name:
#         return "Wrong Lane Driving"
#     else:
#         return "Unknown"

# def update_violations_json():
#     files = [f for f in os.listdir(violations_dir) if os.path.splitext(f)[1].lower() in valid_exts]

#     data = []
#     for idx, f in enumerate(files, start=1):
#         violation_type = detect_violation_type(f)
#         data.append({
#             "track_id": idx,
#             "violation_type": violation_type,
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "image_path": os.path.join(violations_dir, f)
#         })

#     json_path = os.path.join(violations_dir, "violations.json")
#     with open(json_path, "w") as jf:
#         json.dump(data, jf, indent=4)

#     print(f"✅ Updated {len(data)} entries in violations.json")

# if __name__ == "__main__":
#     update_violations_json()
import os
import json
from pathlib import Path
from datetime import datetime
import cv2
import easyocr

# -------------------------------
# Paths
# -------------------------------
violations_dir = Path(r"C:\Users\SURYA TEJA\Desktop\AI_Traffic_Violation_Detector\violations")
violations_dir.mkdir(exist_ok=True)
json_path = violations_dir / "violations.json"

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# -------------------------------
# Collect images and metadata
# -------------------------------
data = []

# Scan all jpg/png images in violations folder
for img_file in violations_dir.glob("*.jpg"):
    try:
        # Load image
        vehicle_crop = cv2.imread(str(img_file))
        if vehicle_crop is None:
            print(f"[WARN] Could not read image: {img_file}")
            continue

        # Run OCR on image
        lp_texts = reader.readtext(vehicle_crop)
        license_plate = lp_texts[0][1] if lp_texts else None

        # Detect violation type from filename (assuming format violationtype_timestamp_id.jpg)
        name_parts = img_file.stem.split("_")
        violation_type = "_".join(name_parts[:-2]) if len(name_parts) >= 3 else "Unknown"

        # Timestamp (from filename or now)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Add to list
        data.append({
            "track_id": name_parts[-1] if len(name_parts) >= 1 else None,
            "violation_type": violation_type,
            "timestamp": timestamp,
            "image_path": str(img_file),
            "clip_path": None,
            "license_plate": license_plate
        })

        print(f"[INFO] Processed {img_file} → Detected type: {violation_type}, LP: {license_plate}")

    except Exception as e:
        print(f"[ERROR] {img_file}: {e}")

# -------------------------------
# Save JSON
# -------------------------------
with open(json_path, "w") as f:
    json.dump(data, f, indent=4)

print(f"[INFO] ✅ Violations JSON updated at {json_path}")
