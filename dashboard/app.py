import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
from streamlit_autorefresh import st_autorefresh

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AI Traffic Violation Dashboard (Advanced)",
    page_icon="🚦",
    layout="wide"
)

# -------------------------------
# Dark Theme Styling
# -------------------------------
st.markdown("""
<style>
    :root { color-scheme: dark; }
    [data-testid="stSidebar"] { background-color: #1E1E1E; }
    .block-container { background-color: #0E1117; color: #FFFFFF; }
    h1,h2,h3 { color: white !important; }
    .stMarkdown p { color: white !important; }
    .violation-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0px 0px 10px rgba(0,255,100,0.15);
        margin-bottom: 20px;
    }
    .chat-box {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 10px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.markdown("<h1 style='text-align:center; color:white;'>🚦 AI Traffic Violation Dashboard (Advanced)</h1>", unsafe_allow_html=True)

# -------------------------------
# JSON Path
# -------------------------------
json_path = r"C:\Users\SURYA TEJA\Desktop\AI_Traffic_Violation_Detector\violations\violations.json"

# -------------------------------
# Helper Function to Load Data
# -------------------------------
def load_data():
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    else:
        st.warning(f"No violations.json file found at: {json_path}")
        return pd.DataFrame()

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("⚙️ Controls")

auto_refresh = st.sidebar.checkbox("Auto-refresh every 10 seconds", value=True, key="auto_refresh")

filter_type = st.sidebar.selectbox(
    "Filter by Violation Type",
    ["All", "signal_jump", "helmet_violation", "overspeeding", "wrong_lane", "triple_riding"],
    index=0,
    key="filter_type"
)

refresh_btn = st.sidebar.button("🔄 Refresh Now")

# -------------------------------
# Chatbot Section
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Violation Chatbot")

user_prompt = st.sidebar.text_input("Ask me something (e.g., 'show red car')", key="chat_input")
send_btn = st.sidebar.button("Send 🚀")

# -------------------------------
# Auto/Manual Refresh
# -------------------------------
if auto_refresh:
    st_autorefresh(interval=10*1000, key="auto_refresh_timer")

if refresh_btn:
    st.experimental_rerun()

# -------------------------------
# Load Data
# -------------------------------
df = load_data()

# -------------------------------
# Chatbot Dummy Response
# -------------------------------
if send_btn and user_prompt:
    st.sidebar.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    st.sidebar.markdown(f"**🧠 You asked:** {user_prompt}")
    st.sidebar.markdown("**🤖 AI Response:** Here's a red four-wheeler car violation detected.")
    st.sidebar.json({
        "track_id": "28",
        "timestamp": "2025-10-26 08:13:19",
        "image_path": r"C:\Users\SURYA TEJA\Desktop\AI_Traffic_Violation_Detector\violations\helmet_violation_1761414922_28.jpg",
        "clip_path": None,
        "license_plate": None,
        "vehicle_color": "Red",
        "vehicle_type": "Four Wheeler"
    })
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Display Violations
# -------------------------------
if not df.empty:
    df = df.sort_values(by="timestamp", ascending=False)

    # Apply filter
    if filter_type != "All":
        df = df[df["violation_type"] == filter_type]

    st.markdown(f"<h2 style='color:#00C853;'>📊 Total Violations: {len(df)}</h2>", unsafe_allow_html=True)

    # Display each violation as a card
    for _, row in df.iterrows():
        with st.container():
            st.markdown('<div class="violation-card">', unsafe_allow_html=True)
            cols = st.columns([1,2])

            # Image
            with cols[0]:
                image_path = row.get("image_path")
                if image_path and Path(image_path).exists():
                    st.image(image_path, caption="Violation Snapshot", use_container_width=True)
                else:
                    st.warning("No image found.")

            # Details
            with cols[1]:
                st.markdown(f"**Violation Type:** {row.get('violation_type','Unknown').replace('_',' ').title()}")
                st.markdown(f"**Track ID:** {row.get('track_id','N/A')}")
                st.markdown(f"**License Plate:** {row.get('license_plate','Unknown')}")
                st.markdown(f"**Timestamp:** {row.get('timestamp','Unknown')}")

                clip_path = row.get("clip_path")
                if clip_path and Path(clip_path).exists():
                    st.video(clip_path)
                else:
                    st.info("No video clip available.")

            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("No violation data available yet. Once detection starts, this dashboard will update automatically.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("<center>Developed with ❤️ for Smart City Traffic Monitoring | © 2025</center>", unsafe_allow_html=True)



# # # import streamlit as st
# # # from pathlib import Path
# # # import os
# # # import cv2
# # # import time
# # # import json
# # # import threading
# # # from PIL import Image
# # # from ultralytics import YOLO
# # # from deep_sort_realtime.deepsort_tracker import DeepSort
# # # import easyocr
# # # import numpy as np

# # # # -------------------------------
# # # # Page Config
# # # # -------------------------------
# # # st.set_page_config(page_title="AI Traffic Violation Dashboard", layout="wide")
# # # st.title("🚦 AI Traffic Violation Dashboard (Demo)")
# # # st.caption("Upload a video, and the AI system will detect traffic violations automatically.")

# # # # -------------------------------
# # # # Upload Video
# # # # -------------------------------
# # # uploaded_video = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])
# # # process_btn = st.button("🚀 Process Video")

# # # # -------------------------------
# # # # Initialize Models
# # # # -------------------------------
# # # @st.cache_resource
# # # def load_models():
# # #     model = YOLO("models/yolov8n.pt")       # Vehicle/person detection
# # #     helmet_model = YOLO("models/helmet.pt") # Helmet detection
# # #     tracker = DeepSort(max_age=30)
# # #     reader = easyocr.Reader(['en'])
# # #     return model, helmet_model, tracker, reader

# # # model, helmet_model, tracker, reader = load_models()

# # # # -------------------------------
# # # # Violations Storage
# # # # -------------------------------
# # # violations_dir = Path("violations")
# # # violations_dir.mkdir(exist_ok=True)
# # # violations_json = violations_dir / "violations.json"
# # # if not violations_json.exists():
# # #     with open(violations_json, "w") as f:
# # #         json.dump([], f, indent=4)

# # # # -------------------------------
# # # # Video Processing Function
# # # # -------------------------------
# # # def process_video(video_path):
# # #     cap = cv2.VideoCapture(video_path)
# # #     frame_count = 0
# # #     all_violations = []
# # #     clip_frames = {}
# # #     prev_centroids = {}
# # #     prev_times = {}
# # #     line_y = 300
# # #     lane_lines = [200, 400]
# # #     speed_limit = 30
# # #     ocr_skip = 5
# # #     helmet_skip = 2

# # #     while True:
# # #         ret, frame = cap.read()
# # #         if not ret:
# # #             break
# # #         frame_count += 1
# # #         h, w = frame.shape[:2]
# # #         frame_time = time.time()

# # #         results = model(frame, imgsz=640)[0]
# # #         detections = []
# # #         for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
# # #             label = int(cls)
# # #             if label in [0,1,2,3,5,7]:
# # #                 x1, y1, x2, y2 = map(int, box)
# # #                 detections.append(([x1, y1, x2, y2], conf, label))

# # #         tracks = tracker.update_tracks(detections, frame=frame)

# # #         # Dummy signal state (for demo)
# # #         signal_state = "RED"

# # #         for track in tracks:
# # #             if not track.is_confirmed():
# # #                 continue
# # #             tid = track.track_id
# # #             x1, y1, x2, y2 = map(int, track.to_ltrb())
# # #             cX, cY = (int((x1+x2)/2), int((y1+y2)/2))

# # #             # Safe crop
# # #             vehicle_crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)].copy()
# # #             if vehicle_crop.size == 0:
# # #                 prev_centroids[tid] = (cX, cY)
# # #                 prev_times[tid] = frame_time
# # #                 continue

# # #             # OCR
# # #             if frame_count % ocr_skip == 0:
# # #                 lp_texts = reader.readtext(vehicle_crop)
# # #                 lp = lp_texts[0][1] if lp_texts else "UNKNOWN"
# # #             else:
# # #                 lp = "UNKNOWN"

# # #             violation_type = None
# # #             helmet_violation = False

# # #             # Example: Signal jump
# # #             if tid in prev_centroids:
# # #                 _, prevY = prev_centroids[tid]
# # #                 if prevY < line_y <= cY and signal_state=="RED":
# # #                     violation_type = "signal_violation"

# # #             # Save violation
# # #             if violation_type:
# # #                 fname = violations_dir / f"{violation_type}_{frame_time:.0f}_{tid}.jpg"
# # #                 cv2.imwrite(str(fname), vehicle_crop)

# # #                 all_violations.append({
# # #                     "violation_type": violation_type,
# # #                     "track_id": tid,
# # #                     "license_plate": lp,
# # #                     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
# # #                     "image_path": str(fname),
# # #                     "clip_path": None,
# # #                     "helmet_violation": helmet_violation
# # #                 })

# # #             prev_centroids[tid] = (cX, cY)
# # #             prev_times[tid] = frame_time

# # #     cap.release()

# # #     # Save JSON
# # #     with open(violations_json, "w") as f:
# # #         json.dump(all_violations, f, indent=4)

# # #     return all_violations

# # # # -------------------------------
# # # # Run Processing
# # # # -------------------------------
# # # if process_btn and uploaded_video is not None:
# # #     temp_file = "temp_video.mp4"
# # #     with open(temp_file, "wb") as f:
# # #         f.write(uploaded_video.getbuffer())
# # #     st.info("Processing video... this may take a while.")
# # #     violations = process_video(temp_file)
# # #     st.success(f"✅ Processing finished! {len(violations)} violations detected.")

# # # # -------------------------------
# # # # Display Violations
# # # # -------------------------------
# # # if violations_json.exists():
# # #     with open(violations_json, "r") as f:
# # #         data = json.load(f)
# # #     if data:
# # #         for ev in reversed(data):
# # #             cols = st.columns([1,2])
# # #             with cols[0]:
# # #                 if Path(ev["image_path"]).exists():
# # #                     st.image(ev["image_path"],  width="stretch")
# # #             with cols[1]:
# # #                 st.markdown(f"""
# # #                 **Violation Type:** {ev.get('violation_type', 'Unknown')}  
# # #                 **Track ID:** {ev.get('track_id', 'Unknown')}  
# # #                 **License Plate:** {ev.get('license_plate', 'Unknown')}  
# # #                 **Timestamp:** {ev.get('timestamp', 'Unknown')}  
# # #                 """)
# # #                 st.divider()
