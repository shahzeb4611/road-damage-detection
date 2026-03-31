import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import io
import tempfile
import pathlib
import numpy as np

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS – dark glassmorphism theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0f0;
    min-height: 100vh;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Cards */
.glass-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 18px;
    backdrop-filter: blur(10px);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #4776e6, #8e54e9);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 15px;
    transition: all 0.3s ease;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(78,115,233,0.45);
}

/* Progress bar */
.stProgress > div > div { background: linear-gradient(90deg, #4776e6, #8e54e9); }

/* Tabs */
button[data-baseweb="tab"] { color: #a0a0c0 !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #8e54e9 !important;
    border-bottom: 3px solid #8e54e9 !important;
}

/* Upload area */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(142,84,233,0.5) !important;
    border-radius: 14px !important;
    padding: 10px !important;
    background: rgba(255,255,255,0.04) !important;
}

h1, h2, h3 { color: #ffffff; }

.damage-tag {
    display: inline-block;
    background: rgba(142,84,233,0.3);
    border: 1px solid #8e54e9;
    color: #d4baff;
    border-radius: 8px;
    padding: 4px 12px;
    margin: 3px;
    font-size: 13px;
    font-weight: 500;
}

.stat-box {
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.12);
}
.stat-number { font-size: 32px; font-weight: 700; color: #8e54e9; }
.stat-label  { font-size: 12px; color: #a0a0c0; }

.video-note {
    background: rgba(71,118,230,0.15);
    border-left: 4px solid #4776e6;
    padding: 10px 16px;
    border-radius: 0 10px 10px 0;
    font-size: 13px;
    color: #c0d0ff;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "road_damage_best.pt")
    if not os.path.exists(model_path):
        return None
    return YOLO(model_path)

model = load_model()

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 10px 0 20px;">
    <h1 style="font-size:42px; font-weight:700; margin:0;">
        🛣️ Road Damage Detection System
    </h1>
    <p style="color:#a0a0c0; font-size:16px; margin-top:8px;">
        AI-powered detection of potholes and cracks in images &amp; videos
    </p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("❌ Model not found at `models/road_damage_best.pt`. Please train or place the model file first.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05,
                               help="Minimum confidence to show a detection")
    iou_threshold  = st.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.45, 0.05,
                               help="Non-max suppression overlap threshold")

    class_options  = list(model.names.values())
    selected_classes = st.multiselect("Filter Classes", class_options, default=class_options)

    st.markdown("---")
    st.markdown("### 📋 Damage Classes")
    st.markdown("""
- 🔴 **Longitudinal Crack** – parallel to traffic  
- 🟠 **Transverse Crack** – perpendicular to traffic  
- 🟡 **Alligator Crack** – interconnected web of cracks  
- 🟣 **Pothole** – depression from erosion & wear  
""")

# ─────────────────────────────────────────────────────────────
# Helper: filter results by selected classes
# ─────────────────────────────────────────────────────────────
def filter_and_plot(result):
    """Remove boxes not in selected_classes, then plot."""
    if selected_classes:
        keep = []
        for i, box in enumerate(result.boxes):
            cls_name = model.names[int(box.cls[0].item())]
            if cls_name in selected_classes:
                keep.append(i)
        # Rebuild result with only kept indices
        if keep:
            result.boxes = result.boxes[keep]
        else:
            result.boxes = result.boxes[0:0]  # empty
    return result.plot()  # BGR


def get_detections_text(result):
    """Return list of (class_name, confidence) tuples."""
    detections = []
    for box in result.boxes:
        cls_name = model.names[int(box.cls[0].item())]
        if not selected_classes or cls_name in selected_classes:
            conf = box.conf[0].item()
            detections.append((cls_name, conf))
    return detections


# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab_img, tab_vid, tab_cam = st.tabs(["🖼️  Images", "🎬  Video", "📷  Webcam"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 – Multiple Image Upload
# ═══════════════════════════════════════════════════════════════
with tab_img:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Images")
    st.markdown("Select **one or multiple** road images. All will be scanned together.")
    uploaded_files = st.file_uploader(
        "Choose road images...",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="multi_image_upload",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        detect_btn = st.button(f"🔍 Detect Damage in {len(uploaded_files)} Image(s)", key="detect_images")

        if detect_btn:
            all_detections = []
            total_damages = 0

            progress = st.progress(0, text="Processing images…")

            for idx, ufile in enumerate(uploaded_files):
                img_pil = Image.open(ufile).convert("RGB")
                img_np  = np.array(img_pil)

                results = model.predict(img_np, conf=conf_threshold, iou=iou_threshold, verbose=False)
                result  = results[0]

                annotated_bgr = filter_and_plot(result)
                annotated_rgb = annotated_bgr[..., ::-1]

                dets = get_detections_text(result)
                total_damages += len(dets)
                all_detections.append((ufile.name, img_pil, annotated_rgb, dets))

                progress.progress((idx + 1) / len(uploaded_files),
                                  text=f"Processing {idx+1}/{len(uploaded_files)}…")

            progress.empty()

            # Summary stats
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="stat-box"><div class="stat-number">{len(uploaded_files)}</div><div class="stat-label">Images Scanned</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="stat-box"><div class="stat-number">{total_damages}</div><div class="stat-label">Damages Found</div></div>', unsafe_allow_html=True)
            dam_imgs = sum(1 for _, _, _, d in all_detections if d)
            c3.markdown(f'<div class="stat-box"><div class="stat-number">{dam_imgs}</div><div class="stat-label">Damaged Images</div></div>', unsafe_allow_html=True)

            st.markdown("---")

            # Show each image result
            for fname, orig_pil, ann_rgb, dets in all_detections:
                with st.expander(f"📷 **{fname}** — {len(dets)} detection(s)", expanded=True):
                    col_orig, col_ann = st.columns(2)
                    with col_orig:
                        st.markdown("**Original**")
                        st.image(orig_pil, use_container_width=True)
                    with col_ann:
                        st.markdown("**Detected Damages**")
                        st.image(ann_rgb, use_container_width=True)

                    if dets:
                        st.markdown("**Detections:**")
                        tags_html = " ".join(
                            f'<span class="damage-tag">{name} · {conf:.0%}</span>'
                            for name, conf in dets
                        )
                        st.markdown(tags_html, unsafe_allow_html=True)
                    else:
                        st.success("✅ No damage detected in this image.")

                    # Download annotated image
                    buf = io.BytesIO()
                    Image.fromarray(ann_rgb).save(buf, format="PNG")
                    buf.seek(0)
                    st.download_button(
                        f"⬇️ Download annotated {fname}",
                        data=buf,
                        file_name=f"annotated_{fname}",
                        mime="image/png",
                        key=f"dl_{fname}_{idx if False else id(dets)}",
                    )

# ═══════════════════════════════════════════════════════════════
# TAB 2 – Video Processing
# ═══════════════════════════════════════════════════════════════
with tab_vid:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Road Video(s)")
    st.markdown("Upload **one or more** videos. Each will be fully processed – every frame annotated.")
    video_files = st.file_uploader(
        "Choose road video(s)...",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
        key="multi_video_upload",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="video-note">
💡 <strong>Tip:</strong> Once processed, the output video player supports <strong>pause, seek, and scrub</strong>.
Stop anywhere to read the crack type label shown in each frame.
</div>
""", unsafe_allow_html=True)

    if video_files:
        process_btn = st.button(f"▶️ Process {len(video_files)} Video(s)", key="process_videos")

        if process_btn:
            for v_idx, vfile in enumerate(video_files):
                st.markdown(f"---\n#### 🎬 Processing: `{vfile.name}`")

                # Save upload to temp file
                suffix = pathlib.Path(vfile.name).suffix
                tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp_in.write(vfile.read())
                tmp_in.flush()
                tmp_in.close()

                cap = cv2.VideoCapture(tmp_in.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Output temp file
                tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_out.close()

                # Try H.264 (avc1) first – browser-compatible; fallback to mp4v
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(tmp_out.name, fourcc, fps, (width, height))
                if not writer.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(tmp_out.name, fourcc, fps, (width, height))

                prog_bar = st.progress(0, text=f"Annotating frames… 0/{total_frames}")
                status_txt = st.empty()

                frame_num = 0
                vid_damages = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
                    result  = results[0]

                    # Count detections for stats
                    for box in result.boxes:
                        cls_name = model.names[int(box.cls[0].item())]
                        if not selected_classes or cls_name in selected_classes:
                            vid_damages += 1

                    annotated = filter_and_plot(result)   # BGR
                    writer.write(annotated)

                    frame_num += 1
                    if frame_num % 10 == 0 or frame_num == total_frames:
                        pct = frame_num / max(total_frames, 1)
                        prog_bar.progress(min(pct, 1.0),
                                          text=f"Annotating frames… {frame_num}/{total_frames}")

                cap.release()
                writer.release()
                prog_bar.empty()
                status_txt.empty()

                # ── Re-encode to H.264 via ffmpeg if available ──────────
                import subprocess, base64
                tmp_h264 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_h264.close()
                try:
                    enc = subprocess.run(
                        ["ffmpeg", "-y", "-i", tmp_out.name,
                         "-vcodec", "libx264", "-crf", "23",
                         "-pix_fmt", "yuv420p", tmp_h264.name],
                        capture_output=True, timeout=300
                    )
                    video_path = tmp_h264.name if enc.returncode == 0 else tmp_out.name
                except Exception:
                    video_path = tmp_out.name

                with open(video_path, "rb") as f:
                    video_bytes = f.read()

                # Stats row
                c1, c2, c3 = st.columns(3)
                c1.markdown(f'<div class="stat-box"><div class="stat-number">{frame_num}</div><div class="stat-label">Frames Processed</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="stat-box"><div class="stat-number">{vid_damages}</div><div class="stat-label">Detections Across All Frames</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="stat-box"><div class="stat-number">{fps:.0f} fps</div><div class="stat-label">Video Frame Rate</div></div>', unsafe_allow_html=True)

                st.markdown("#### 🎥 Annotated Output Video")

                # ── Custom HTML5 player (base64 embedded, plays in browser) ──
                b64_video = base64.b64encode(video_bytes).decode()
                unique_id  = f"vid_{v_idx}"
                player_html = f"""
<style>
  .player-wrapper-{unique_id} {{
    background: rgba(0,0,0,0.55);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 14px;
    margin-top: 10px;
  }}
  .player-wrapper-{unique_id} video {{
    width: 100%;
    border-radius: 10px;
    outline: none;
  }}
  .speed-bar-{unique_id} {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 12px;
    flex-wrap: wrap;
  }}
  .speed-bar-{unique_id} label {{
    color: #c0c0e0;
    font-family: Inter, sans-serif;
    font-size: 13px;
    font-weight: 600;
    white-space: nowrap;
  }}
  .speed-bar-{unique_id} input[type=range] {{
    flex: 1;
    min-width: 160px;
    accent-color: #8e54e9;
  }}
  .speed-badge-{unique_id} {{
    background: linear-gradient(90deg,#4776e6,#8e54e9);
    color: white;
    font-family: Inter, sans-serif;
    font-size: 13px;
    font-weight: 700;
    border-radius: 8px;
    padding: 4px 14px;
    white-space: nowrap;
  }}
  .speed-btn-row-{unique_id} {{
    display: flex;
    gap: 6px;
    margin-top: 8px;
    flex-wrap: wrap;
  }}
  .speed-btn-{unique_id} {{
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    color: #d0d0f0;
    border-radius: 7px;
    padding: 4px 12px;
    cursor: pointer;
    font-size: 12px;
    font-family: Inter, sans-serif;
    transition: background 0.2s;
  }}
  .speed-btn-{unique_id}:hover, .speed-btn-{unique_id}.active-btn {{
    background: linear-gradient(90deg,#4776e6,#8e54e9);
    color: white;
    border-color: transparent;
  }}
</style>

<div class="player-wrapper-{unique_id}">
  <video id="{unique_id}" controls>
    <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
    Your browser does not support HTML5 video.
  </video>

  <div class="speed-bar-{unique_id}">
    <label>⏩ Playback Speed:</label>
    <input type="range" id="slider_{unique_id}" min="0.1" max="4" step="0.05" value="1"
      oninput="
        var v=parseFloat(this.value);
        document.getElementById('{unique_id}').playbackRate=v;
        document.getElementById('badge_{unique_id}').innerText=v.toFixed(2)+'x';
      ">
    <span class="speed-badge-{unique_id}" id="badge_{unique_id}">1.00x</span>
  </div>

  <div class="speed-btn-row-{unique_id}">
    <span style="color:#a0a0c0;font-size:12px;font-family:Inter,sans-serif;align-self:center;">Quick select:</span>
    {''.join(f'''<button class="speed-btn-{unique_id}" onclick="
      var spd={spd};
      document.getElementById('{unique_id}').playbackRate=spd;
      document.getElementById('slider_{unique_id}').value=spd;
      document.getElementById('badge_{unique_id}').innerText=spd.toFixed(2)+'x';
    ">{label}</button>''' for spd, label in [(0.25,'0.25x'),(0.5,'0.5x'),(0.75,'0.75x'),(1.0,'1x'),(1.25,'1.25x'),(1.5,'1.5x'),(2.0,'2x'),(3.0,'3x'),(4.0,'4x')])}
  </div>
</div>
"""
                import streamlit.components.v1 as components
                components.html(player_html, height=600, scrolling=False)

                # Download button
                st.download_button(
                    label=f"⬇️ Download annotated video – {vfile.name}",
                    data=video_bytes,
                    file_name=f"annotated_{vfile.name}",
                    mime="video/mp4",
                    key=f"dl_vid_{v_idx}",
                )

                # Cleanup temp files
                try:
                    os.unlink(tmp_in.name)
                    os.unlink(tmp_out.name)
                    if video_path != tmp_out.name:
                        os.unlink(tmp_h264.name)
                except Exception:
                    pass

# ═══════════════════════════════════════════════════════════════
# TAB 3 – Live Webcam
# ═══════════════════════════════════════════════════════════════
with tab_cam:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📷 Live Webcam Detection")
    st.markdown("Toggle the checkbox to start/stop the live feed. Detections are overlaid in real-time.")
    st.markdown('</div>', unsafe_allow_html=True)

    run_webcam = st.checkbox("▶️ Start Webcam Detection", key="webcam_active")

    if run_webcam:
        st.info("Webcam initialising… Uncheck the box above to stop.")
        cap = cv2.VideoCapture(0)
        placeholder = st.empty()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Could not access webcam. Make sure it is not used by another app.")
                    break

                results = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
                annotated = filter_and_plot(results[0])[..., ::-1]   # BGR→RGB
                placeholder.image(annotated, channels="RGB", use_container_width=True)
        finally:
            cap.release()
            cv2.destroyAllWindows()
