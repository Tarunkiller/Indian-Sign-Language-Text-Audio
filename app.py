"""
Indian Sign Language → Text & Audio
======================================
Streamlit application with:
 • Live webcam capture (via st.camera_input)
 • MediaPipe hand-landmark detection
 • MLP gesture classifier (trained on synthetic ISL data)
 • Sentence builder with hold-to-confirm logic
 • Text-to-speech via gTTS (browser audio)
"""

import os, sys, time, io, pathlib
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# ── Page config (MUST be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="ISL → Text & Audio",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = pathlib.Path(__file__).parent
MODEL_PATH  = ROOT / "model" / "isl_model.pkl"
ENC_PATH    = ROOT / "model" / "label_encoder.pkl"
SCL_PATH    = ROOT / "model" / "scaler.pkl"

# ── Premium CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: #0a0e1a;
    color: #e2e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #111827 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Header Banner ── */
.isl-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4c1d95 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    border: 1px solid rgba(129,140,248,0.3);
    box-shadow: 0 8px 32px rgba(99,102,241,0.25);
    display: flex;
    align-items: center;
    gap: 20px;
}
.isl-header h1 {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a5b4fc, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.isl-header p {
    color: #94a3b8;
    margin: 4px 0 0;
    font-size: 0.95rem;
}
.isl-emoji { font-size: 3rem; }

/* ── Cards ── */
.card {
    background: rgba(15,22,41,0.85);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
    transition: border-color 0.3s;
}
.card:hover { border-color: rgba(129,140,248,0.5); }

/* ── Prediction Badge ── */
.pred-badge {
    font-size: 5rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    padding: 10px 0;
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.75; }
}
.pred-label {
    text-align: center;
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: -4px;
}

/* ── Sentence Display ── */
.sentence-box {
    background: linear-gradient(135deg, rgba(30,27,75,0.6), rgba(49,46,129,0.4));
    border: 1px solid rgba(129,140,248,0.35);
    border-radius: 12px;
    padding: 18px 22px;
    font-size: 1.35rem;
    font-weight: 600;
    color: #e2e8f0;
    letter-spacing: 0.04em;
    min-height: 64px;
    word-break: break-all;
}

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: rgba(30,41,59,0.8);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    margin-top: 8px;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    transition: width 0.4s ease;
}

/* ── Status dots ── */
.status-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.dot-green  { background:#22c55e; box-shadow:0 0 6px #22c55e88; animation: blink 1.2s infinite; }
.dot-red    { background:#ef4444; }
.dot-yellow { background:#f59e0b; box-shadow:0 0 6px #f59e0b88; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── ISL Reference grid ── */
.ref-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(48px, 1fr));
    gap: 8px;
    margin-top: 12px;
}
.ref-cell {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 8px;
    text-align: center;
    padding: 6px 4px;
    font-weight: 700;
    font-size: 1rem;
    color: #a5b4fc;
    cursor: default;
    transition: background 0.2s;
}
.ref-cell:hover { background: rgba(99,102,241,0.35); }

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    transition: opacity 0.2s !important;
    width: 100%;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }

/* ── Slider ── */
[data-testid="stSlider"] .thumb { background: #6366f1 !important; }
[data-baseweb="slider"] .track { background: rgba(99,102,241,0.3) !important; }

/* ── Section titles ── */
.sec-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load trained MLP, label encoder, and scaler. Train if not found."""
    if not MODEL_PATH.exists():
        return None, None, None
    import joblib
    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENC_PATH)
    scaler  = joblib.load(SCL_PATH)
    return model, encoder, scaler


@st.cache_resource(show_spinner=False)
def get_hands():
    from utils.landmark_utils import get_hand_landmarker
    return get_hand_landmarker()


# ── Core inference ────────────────────────────────────────────────────────────
def predict_from_frame(frame_bgr, model, encoder, scaler, hands):
    from utils.landmark_utils import extract_landmarks, draw_landmarks, get_hand_bbox

    landmarks, detected, results = extract_landmarks(frame_bgr, hands)
    annotated = draw_landmarks(frame_bgr.copy(), results)

    if not detected or landmarks is None:
        return annotated, None, 0.0, None

    feat = scaler.transform([landmarks])
    proba = model.predict_proba(feat)[0]
    idx = int(np.argmax(proba))
    confidence = float(proba[idx])
    prediction = encoder.classes_[idx]

    bbox = get_hand_bbox(frame_bgr, results)
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (99, 102, 241), 2)
        cv2.putText(annotated, f"{prediction}  {confidence*100:.0f}%",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (165, 180, 252), 2)

    return annotated, prediction, confidence, bbox


# ── TTS helper ────────────────────────────────────────────────────────────────
def generate_audio(text: str):
    from utils.tts_utils import speak_gtts
    return speak_gtts(text)


# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "sentence": "",
        "last_pred": None,
        "hold_counter": 0,
        "history": [],
        "audio_bytes": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🤟 ISL Translator")
    st.markdown("---")

    st.markdown('<div class="sec-title">⚙️ Recognition Settings</div>', unsafe_allow_html=True)
    conf_thresh   = st.slider("Confidence threshold", 0.30, 0.95, 0.55, 0.05,
                              help="Minimum confidence to count a prediction")
    hold_frames   = st.slider("Hold frames to confirm", 5, 40, 18, 1,
                              help="How many captures must agree before adding a letter")

    st.markdown("---")
    st.markdown('<div class="sec-title">🔊 Speech Settings</div>', unsafe_allow_html=True)
    use_tts = st.toggle("Enable text-to-speech", value=True)

    st.markdown("---")
    st.markdown('<div class="sec-title">🗑️ Controls</div>', unsafe_allow_html=True)
    if st.button("🗑️  Clear Sentence"):
        st.session_state.sentence = ""
        st.session_state.audio_bytes = None
        st.session_state.last_pred = None
        st.session_state.hold_counter = 0

    if st.button("⌫  Delete Last Char"):
        st.session_state.sentence = st.session_state.sentence[:-1]

    if st.button("␣  Add Space"):
        st.session_state.sentence += " "

    st.markdown("---")

    # ISL reference grid
    st.markdown('<div class="sec-title">📖 ISL Quick Reference</div>', unsafe_allow_html=True)
    classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("0123456789")
    cells   = "".join(f'<div class="ref-cell">{c}</div>' for c in classes)
    st.markdown(f'<div class="ref-grid">{cells}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.72rem;color:#475569;text-align:center">'
        'ISL → Text & Audio • Deep Learning Demo</p>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="isl-header">
  <div class="isl-emoji">🤟</div>
  <div>
    <h1>Indian Sign Language → Text & Audio</h1>
    <p>AI-powered gesture recognition • Point your hand at the camera and hold each sign to build a sentence</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
model, encoder, scaler = load_model()
hands = get_hands() if model is not None else None

model_ok = model is not None

# Model status banner
if not model_ok:
    st.warning(
        "⚠️ **Model not trained yet.** "
        "Run `python model/train_model.py` from the project root, then reload this page.",
        icon="⚠️"
    )

# ── Layout: camera | prediction | sentence ────────────────────────────────────
col_cam, col_right = st.columns([3, 2], gap="large")

with col_cam:
    st.markdown('<div class="sec-title">📷 Camera Input</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.82rem;color:#64748b;margin-top:-6px;margin-bottom:10px">'
        'Click <b>Take Photo</b> to capture a gesture, or keep clicking for continuous recognition.</p>',
        unsafe_allow_html=True
    )

    camera_img = st.camera_input(
        label="",
        key="cam",
        label_visibility="collapsed",
    )

    # Annotated image placeholder
    frame_placeholder = st.empty()

with col_right:
    # ── Prediction card ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">🔍 Current Prediction</div>', unsafe_allow_html=True)
    pred_display    = st.empty()
    conf_bar_display = st.empty()
    status_display  = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sentence card ────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">📝 Sentence Builder</div>', unsafe_allow_html=True)
    sentence_display = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TTS ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">🔊 Text-to-Speech</div>', unsafe_allow_html=True)
    tts_col1, tts_col2 = st.columns(2)
    with tts_col1:
        speak_btn = st.button("▶️  Speak Sentence", disabled=not use_tts)
    with tts_col2:
        copy_display = st.empty()

    audio_placeholder = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Recent History ────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">🕑 Recognition History</div>', unsafe_allow_html=True)
    history_display = st.empty()


# ── Process captured frame ────────────────────────────────────────────────────
current_pred  = None
current_conf  = 0.0

if camera_img is not None and model_ok:
    # Decode image
    img_bytes = camera_img.getvalue()
    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Run inference
    annotated, prediction, confidence, bbox = predict_from_frame(
        frame_bgr, model, encoder, scaler, hands
    )
    current_pred = prediction
    current_conf = confidence

    # Show annotated frame
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

    # ── Hold-to-confirm logic ─────────────────────────────────────────────────
    if prediction and confidence >= conf_thresh:
        if prediction == st.session_state.last_pred:
            st.session_state.hold_counter += 1
            if st.session_state.hold_counter >= hold_frames:
                # Commit letter
                st.session_state.sentence += prediction
                st.session_state.history.append(
                    {"sign": prediction, "conf": f"{confidence*100:.0f}%"}
                )
                # Keep history last 15
                st.session_state.history = st.session_state.history[-15:]
                st.session_state.hold_counter = 0
                st.session_state.last_pred = None
        else:
            st.session_state.last_pred    = prediction
            st.session_state.hold_counter = 1
    else:
        st.session_state.last_pred    = None
        st.session_state.hold_counter = 0

elif camera_img is None:
    frame_placeholder.markdown(
        '<div class="card" style="text-align:center;padding:60px 20px;">'
        '<div style="font-size:3rem">📷</div>'
        '<p style="color:#64748b;margin-top:12px">Point your hand at the camera<br>and click <b>Take Photo</b></p>'
        '</div>',
        unsafe_allow_html=True
    )

# ── Render prediction card ─────────────────────────────────────────────────────
hold_pct = (st.session_state.hold_counter / max(hold_frames, 1)) * 100

if current_pred:
    dot_cls = "dot-green"
    status_txt = f"Detecting • hold {st.session_state.hold_counter}/{hold_frames}"
    pred_display.markdown(
        f'<div class="card">'
        f'<div class="pred-badge">{current_pred}</div>'
        f'<div class="pred-label">Detected Sign</div>'
        f'<div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{current_conf*100:.0f}%"></div></div>'
        f'<p style="text-align:center;margin:6px 0 0;font-size:0.78rem;color:#94a3b8">'
        f'Confidence: {current_conf*100:.1f}%</p>'
        f'</div>',
        unsafe_allow_html=True
    )
    conf_bar_display.markdown(
        f'<div style="margin-top:8px">'
        f'<p style="font-size:0.75rem;color:#64748b;margin:0">Hold progress</p>'
        f'<div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{hold_pct:.0f}%;background:linear-gradient(90deg,#22c55e,#4ade80)"></div></div>'
        f'</div>',
        unsafe_allow_html=True
    )
else:
    dot_cls = "dot-yellow" if camera_img is not None else "dot-red"
    pred_display.markdown(
        '<div class="card" style="text-align:center;padding:30px 20px;">'
        '<div style="font-size:2.5rem;color:#475569">—</div>'
        '<div class="pred-label" style="margin-top:8px">No hand detected</div>'
        '</div>',
        unsafe_allow_html=True
    )
    conf_bar_display.empty()

status_display.markdown(
    f'<p style="font-size:0.78rem;color:#64748b;margin-top:6px">'
    f'<span class="status-dot {dot_cls}"></span>'
    f'{"Recognizing" if current_pred else "Waiting for hand"}'
    f'</p>',
    unsafe_allow_html=True
)

# ── Render sentence ────────────────────────────────────────────────────────────
sentence = st.session_state.sentence or ""
sentence_display.markdown(
    f'<div class="sentence-box">{sentence if sentence else "<span style=\'color:#374151\'>Your sentence will appear here…</span>"}</div>',
    unsafe_allow_html=True
)

# ── Speak ─────────────────────────────────────────────────────────────────────
if speak_btn and sentence.strip():
    with st.spinner("Generating audio …"):
        audio_bytes = generate_audio(sentence.strip())
    if audio_bytes:
        st.session_state.audio_bytes = audio_bytes

if st.session_state.audio_bytes:
    audio_placeholder.audio(st.session_state.audio_bytes, format="audio/mp3")

# ── History ───────────────────────────────────────────────────────────────────
history = st.session_state.history
if history:
    rows = "".join(
        f'<tr>'
        f'<td style="padding:4px 10px;font-weight:700;color:#a5b4fc">{h["sign"]}</td>'
        f'<td style="padding:4px 10px;color:#94a3b8">{h["conf"]}</td>'
        f'</tr>'
        for h in reversed(history[-8:])
    )
    history_display.markdown(
        f'<div class="card" style="padding:12px 16px">'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<tr><th style="text-align:left;color:#6366f1;font-size:0.72rem;padding:4px 10px">SIGN</th>'
        f'<th style="text-align:left;color:#6366f1;font-size:0.72rem;padding:4px 10px">CONF</th></tr>'
        f'{rows}'
        f'</table></div>',
        unsafe_allow_html=True
    )
else:
    history_display.markdown(
        '<div class="card" style="text-align:center;color:#374151;font-size:0.82rem;padding:16px">No recognitions yet</div>',
        unsafe_allow_html=True
    )
