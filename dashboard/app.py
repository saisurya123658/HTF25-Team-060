import streamlit as st
import pandas as pd
import json
import os
import cv2
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
import easyocr
from streamlit_autorefresh import st_autorefresh

# -----------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(
    page_title="AI Traffic Violation Dashboard (Advanced)",
    page_icon="🚦",
    layout="wide"
)

# -----------------------------------------------------------
# DARK THEME STYLING
# -----------------------------------------------------------
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

# -----------------------------------------------------------
# TITLE
# -----------------------------------------------------------
st.markdown("<h1 style='text-align:center; color:white;'>🚦 AI Traffic Violation Dashboard (Advanced)</h1>", unsafe_allow_html=True)

# -----------------------------------------------------------
# PATHS
# -----------------------------------------------------------
violations_dir = Path(r"C:\Users\SURYA TEJA\Desktop\AI_Traffic_Violation_Detector\violations")
violations_dir.mkdir(exist_ok=True)
violations_json = violations_dir / "violations.json"
if not violations_json.exists():
    with open(violations_json, "w") as f:
        json.dump([], f, indent=4)

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------
def load_data():
    if violations_json.exists():
        with open(violations_json, "r") as f:
            data = json.load(f)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    else:
        st.warning("No violations.json file found.")
        return pd.DataFrame()

# -----------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------
st.sidebar.header("⚙️ Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh every 10 seconds", value=True, key="auto_refresh")
filter_type = st.sidebar.selectbox(
    "Filter by Violation Type",
    ["All", "signal_jump", "helmet_violation", "overspeeding", "wrong_lane", "triple_riding"],
    index=0
)
refresh_btn = st.sidebar.button("🔄 Refresh Now")

# Chatbot
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Violation Chatbot")
user_prompt = st.sidebar.text_input("Ask me something (e.g., 'show helmet violations')")
send_btn = st.sidebar.button("Send 🚀")

# -----------------------------------------------------------
# AUTO REFRESH
# -----------------------------------------------------------
if auto_refresh:
    st_autorefresh(interval=10 * 1000, key="auto_refresh_timer")

if refresh_btn:
    st.experimental_rerun()

# -----------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------
@st.cache_resource
def load_models():
    model = YOLO("models/yolov8n.pt")
    helmet_model = YOLO("models/helmet.pt")
    tracker = DeepSort(max_age=30)
    reader = easyocr.Reader(['en'])
    return model, helmet_model, tracker, reader

model, helmet_model, tracker, reader = load_models()

# -----------------------------------------------------------
# VIDEO UPLOAD & PROCESSING
# -----------------------------------------------------------
st.markdown("## 🎥 Upload and Process Traffic Video")
uploaded_video = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])
process_btn = st.button("🚀 Process Video")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    all_violations = []
    prev_centroids = {}
    line_y = 300
    speed_limit = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        h, w = frame.shape[:2]
        frame_time = time.time()

        results = model(frame, imgsz=640)[0]
        detections = []
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            label = int(cls)
            if label in [0, 1, 2, 3, 5, 7]:  # vehicles
                x1, y1, x2, y2 = map(int, box)
                detections.append(([x1, y1, x2, y2], conf, label))

        tracks = tracker.update_tracks(detections, frame=frame)

        signal_state = "RED"  # dummy for demo
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cX, cY = (int((x1+x2)/2), int((y1+y2)/2))
            vehicle_crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)].copy()

            if frame_count % 5 == 0:
                lp_texts = reader.readtext(vehicle_crop)
                lp = lp_texts[0][1] if lp_texts else "UNKNOWN"
            else:
                lp = "UNKNOWN"

            violation_type = None
            if tid in prev_centroids:
                _, prevY = prev_centroids[tid]
                if prevY < line_y <= cY and signal_state == "RED":
                    violation_type = "signal_jump"

            if violation_type:
                fname = violations_dir / f"{violation_type}_{int(frame_time)}_{tid}.jpg"
                cv2.imwrite(str(fname), vehicle_crop)
                all_violations.append({
                    "violation_type": violation_type,
                    "track_id": tid,
                    "license_plate": lp,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_path": str(fname),
                    "clip_path": None
                })
            prev_centroids[tid] = (cX, cY)

    cap.release()

    # Save JSON
    with open(violations_json, "w") as f:
        json.dump(all_violations, f, indent=4)

    return all_violations

if process_btn and uploaded_video:
    temp_path = "uploaded_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    st.info("Processing video... please wait ⏳")
    violations = process_video(temp_path)
    st.success(f"✅ Done! {len(violations)} violations detected and saved.")

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
df = load_data()

# -----------------------------------------------------------
# CHATBOT RESPONSE (dummy)
# -----------------------------------------------------------
if send_btn and user_prompt:
    st.sidebar.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    st.sidebar.markdown(f"**🧠 You asked:** {user_prompt}")
    st.sidebar.markdown("**🤖 AI Response:** Here's a red four-wheeler violation detected.")
    st.sidebar.json({
        "track_id": "28",
        "timestamp": "2025-10-26 08:13:19",
        "image_path": str(violations_dir / "helmet_violation_1761414922_28.jpg"),
        "vehicle_color": "Red",
        "vehicle_type": "Four Wheeler"
    })
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# DISPLAY VIOLATIONS
# -----------------------------------------------------------
if not df.empty:
    df = df.sort_values(by="timestamp", ascending=False)
    if filter_type != "All":
        df = df[df["violation_type"] == filter_type]

    st.markdown(f"<h2 style='color:#00C853;'>📊 Total Violations: {len(df)}</h2>", unsafe_allow_html=True)

    for _, row in df.iterrows():
        with st.container():
            st.markdown('<div class="violation-card">', unsafe_allow_html=True)
            cols = st.columns([1,2])

            with cols[0]:
                image_path = row.get("image_path")
                if image_path and Path(image_path).exists():
                    st.image(image_path, caption="Violation Snapshot", use_container_width=True)
                else:
                    st.warning("No image found.")

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

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("---")
st.markdown("<center>Developed with ❤️ for Smart City Traffic Monitoring | © 2025</center>", unsafe_allow_html=True)
