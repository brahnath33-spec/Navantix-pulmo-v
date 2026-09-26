import streamlit as st
import numpy as np
import torch
import torchvision
import skimage
from PIL import Image
import torchxrayvision as xrv
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from datetime import datetime
import time
import csv
import io

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Navantix Pulmo",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CLINICAL PALETTE + STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'IBM Plex Sans', -apple-system, sans-serif;
        color: #1a1f2e;
        background: #f5f7fa;
    }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stToolbar"] {display: none;}

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0a1628;
        border-right: 1px solid #1a2740;
    }
    [data-testid="stSidebar"] * {
        color: #cbd5e1;
    }
    [data-testid="stSidebar"] .stRadio > label {
        display: none;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        padding: 0.7rem 1rem;
        border-radius: 4px;
        margin: 0.15rem 0;
        cursor: pointer;
        transition: all 0.15s;
        font-size: 0.92rem;
        font-weight: 400;
        color: #94a3b8;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background: #14213a;
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
        background: #1a2f52;
        color: #ffffff;
        font-weight: 500;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    /* Main content area */
    .main .block-container {
        max-width: 1400px;
        padding: 1.5rem 2.5rem 3rem 2.5rem;
        background: #f5f7fa;
    }

    /* Brand block in sidebar */
    .nx-brand-block {
        padding: 1.5rem 1rem 1.2rem 1rem;
        border-bottom: 1px solid #1a2740;
        margin-bottom: 0.8rem;
    }
    .nx-brand-mark {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 0.02em;
        line-height: 1;
    }
    .nx-brand-mark span { color: #3b82f6; }
    .nx-brand-sub {
        font-size: 0.62rem;
        color: #64748b;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-top: 0.4rem;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Top bar */
    .nx-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.75rem;
    }
    .nx-breadcrumb {
        font-size: 0.85rem;
        color: #64748b;
    }
    .nx-breadcrumb strong {
        color: #0a1628;
        font-weight: 600;
    }
    .nx-status {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.9rem;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 999px;
        font-size: 0.75rem;
        color: #047857;
        font-weight: 500;
    }
    .nx-status-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #10b981;
    }

    /* Page heading */
    .nx-h1 {
        font-size: 1.65rem;
        font-weight: 600;
        color: #0a1628;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.01em;
    }
    .nx-h1-sub {
        font-size: 0.92rem;
        color: #64748b;
        margin-bottom: 2rem;
    }

    /* Cards */
    .nx-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1rem;
    }
    .nx-card-title {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 1rem;
    }

    /* Severity labels */
    .nx-sev {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        padding: 0.2rem 0.55rem;
        border-radius: 3px;
        text-transform: uppercase;
        font-family: 'IBM Plex Mono', monospace;
    }
    .nx-sev-high { background: #fee2e2; color: #991b1b; }
    .nx-sev-mod  { background: #fef3c7; color: #92400e; }
    .nx-sev-low  { background: #e2e8f0; color: #475569; }
    .nx-sev-norm { background: #d1fae5; color: #065f46; }

    /* Result rows */
    .nx-row {
        display: grid;
        grid-template-columns: 220px 1fr 90px 70px;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid #eef2f7;
        font-size: 0.92rem;
    }
    .nx-row:last-child { border-bottom: none; }
    .nx-row-label {
        color: #1e293b;
        font-weight: 400;
    }
    .nx-row-bar {
        height: 6px;
        background: #eef2f7;
        border-radius: 3px;
        overflow: hidden;
        position: relative;
    }
    .nx-row-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.4s;
    }
    .nx-bar-high { background: linear-gradient(90deg, #dc2626, #ef4444); }
    .nx-bar-mod  { background: linear-gradient(90deg, #d97706, #f59e0b); }
    .nx-bar-low  { background: linear-gradient(90deg, #64748b, #94a3b8); }
    .nx-row-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        font-weight: 500;
        text-align: right;
        color: #0a1628;
    }
    .nx-row-sev { text-align: right; }

    /* Decision banner */
    .nx-banner {
        padding: 1.1rem 1.4rem;
        border-radius: 8px;
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
        font-weight: 500;
    }
    .nx-banner-clear {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
    }
    .nx-banner-flag {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
    }
    .nx-banner-icon {
        font-size: 1.3rem;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Metadata chips */
    .nx-meta-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .nx-chip {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        padding: 0.3rem 0.7rem;
        background: #eef2f7;
        border-radius: 3px;
        color: #475569;
        letter-spacing: 0.02em;
    }
    .nx-chip-strong {
        background: #0a1628;
        color: #ffffff;
    }

    /* Buttons */
    .stButton > button {
        background: #0a1628;
        color: #ffffff;
        border: none;
        border-radius: 5px;
        padding: 0.65rem 1.4rem;
        font-weight: 500;
        font-size: 0.88rem;
        font-family: 'IBM Plex Sans', sans-serif;
        letter-spacing: 0.02em;
        transition: all 0.15s;
        width: 100%;
    }
    .stButton > button:hover {
        background: #1e3a8a;
        color: #ffffff;
    }
    .stButton > button:disabled {
        background: #cbd5e1;
        color: #94a3b8;
    }

    /* Primary accent button */
    .stButton > button[kind="primary"] {
        background: #2563eb;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 1.5px dashed #cbd5e1;
        border-radius: 8px;
        background: #ffffff;
        padding: 1rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6;
        background: #f8fafc;
    }

    /* Footer */
    .nx-footer {
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
        padding-top: 1.2rem;
        font-size: 0.7rem;
        color: #94a3b8;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.04em;
        display: flex;
        justify-content: space-between;
    }

    /* Metric cards */
    .nx-metric {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
    }
    .nx-metric-label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .nx-metric-value {
        font-size: 1.6rem;
        font-weight: 600;
        color: #0a1628;
        font-family: 'IBM Plex Mono', monospace;
    }
    .nx-metric-value small {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-left: 0.3rem;
        font-weight: 400;
    }

    /* Table (worklist/reports) */
    .nx-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .nx-table thead th {
        text-align: left;
        padding: 0.7rem 0.9rem;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        border-bottom: 1px solid #e2e8f0;
        background: #f8fafc;
    }
    .nx-table tbody td {
        padding: 0.75rem 0.9rem;
        border-bottom: 1px solid #eef2f7;
        color: #1e293b;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
    }
    .nx-table tbody tr:hover {
        background: #f8fafc;
    }

    /* Slider styling */
    [data-testid="stSlider"] label {
        font-size: 0.82rem;
        color: #475569;
        font-weight: 500;
    }

    /* Radio in sidebar — hide default radio circle */
    [data-testid="stSidebar"] input[type="radio"] { display: none; }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner="Loading Navantix Pulmo model...")
def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model

model = load_model()


# ============================================================
# HELPERS
# ============================================================
def prepare_xray(img_array):
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        try:
            img_array = skimage.color.rgb2gray(img_array)
        except Exception:
            img_array = (0.299 * img_array[:, :, 0]
                         + 0.587 * img_array[:, :, 1]
                         + 0.114 * img_array[:, :, 2])

    img_array = img_array.astype(np.float32)
    mx = float(img_array.max())
    mn = float(img_array.min())
    if mx <= 1.0:
        img_array = img_array * 255.0
    elif mx > 255.0:
        img_array = (img_array - mn) / (mx - mn) * 255.0

    img = xrv.datasets.normalize(img_array, 255)
    if len(img.shape) > 2:
        img = img[:, :, 0] if img.shape[2] == 1 else img.mean(axis=2)
    img = img[None, ...]

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img = transform(img)
    return torch.from_numpy(img)


def run_inference(input_image):
    """Run full pipeline. Returns dict with scores, pairs, timing, cam."""
    t0 = time.time()
    img_array = np.array(input_image.convert("L"))
    img_tensor = prepare_xray(img_array)

    with torch.no_grad():
        outputs = model(img_tensor[None, ...])
    scores = outputs[0].numpy()

    pairs = sorted(zip(model.pathologies, scores), key=lambda x: -x[1])

    # Grad-CAM on top finding
    cam_overlay = None
    try:
        top_name, _ = pairs[0]
        top_idx = model.pathologies.index(top_name)
        target_layer = [model.features.norm5]
        cam = GradCAM(model=model, target_layers=target_layer)
        grayscale_cam = cam(
            input_tensor=img_tensor[None, ...],
            targets=[ClassifierOutputTarget(top_idx)],
        )[0]
        display_img = np.array(input_image.convert("L").resize((224, 224))) / 255.0
        display_img = np.stack([display_img] * 3, axis=-1).astype(np.float32)
        cam_overlay = show_cam_on_image(display_img, grayscale_cam, use_rgb=True)
    except Exception:
        cam_overlay = None

    elapsed_ms = (time.time() - t0) * 1000.0
    return {
        "pairs": pairs,
        "cam": cam_overlay,
        "elapsed_ms": elapsed_ms,
        "top_name": pairs[0][0],
        "top_score": float(pairs[0][1]),
    }


def severity_of(score, threshold):
    if score >= threshold + 0.1:
        return "high"
    if score >= threshold:
        return "mod"
    if score >= threshold * 0.5:
        return "low"
    return "low"


def severity_label(sev):
    return {"high": "HIGH", "mod": "MODERATE", "low": "LOW"}[sev]


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("""
        <div class="nx-brand-block">
            <div class="nx-brand-mark">N<span>+</span></div>
            <div class="nx-brand-sub">Navantix Pulmo<br>Clinical v1.0</div>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Dashboard", "New Study", "Worklist", "Reports", "Settings"],
        label_visibility="collapsed",
    )

    st.markdown("""
        <div style="position:fixed; bottom:1rem; left:1rem; right:1rem;
                    padding-top:1rem; border-top:1px solid #1a2740;
                    font-size:0.68rem; color:#64748b;
                    font-family:'IBM Plex Mono', monospace;
                    letter-spacing:0.04em;">
            ADMIN@NAVANTIXPULMO.LOCAL<br>
            <span style="color:#3b82f6;">●</span> CONNECTED
        </div>
    """, unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================
if "studies" not in st.session_state:
    st.session_state.studies = []
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5
if "active_result" not in st.session_state:
    st.session_state.active_result = None
if "active_image" not in st.session_state:
    st.session_state.active_image = None
if "active_filename" not in st.session_state:
    st.session_state.active_filename = None
if "active_filesize" not in st.session_state:
    st.session_state.active_filesize = None


# ============================================================
# TOP BAR (common to all pages)
# ============================================================
def topbar(page_name):
    st.markdown(f"""
        <div class="nx-topbar">
            <div class="nx-breadcrumb">
                <strong>Navantix Pulmo</strong> &nbsp;/&nbsp; {page_name}
            </div>
            <div class="nx-status">
                <span class="nx-status-dot"></span> Connected
            </div>
        </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE: DASHBOARD
# ============================================================
if page == "Dashboard":
    topbar("Dashboard")

    st.markdown('<div class="nx-h1">Clinical Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-h1-sub">Session overview and recent analysis activity.</div>', unsafe_allow_html=True)

    n_studies = len(st.session_state.studies)
    n_flagged = sum(1 for s in st.session_state.studies if s["flagged"])
    avg_time = np.mean([s["elapsed_ms"] for s in st.session_state.studies]) if n_studies else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
            <div class="nx-metric">
                <div class="nx-metric-label">Studies Analyzed</div>
                <div class="nx-metric-value">{n_studies}</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="nx-metric">
                <div class="nx-metric-label">Flagged Studies</div>
                <div class="nx-metric-value">{n_flagged}</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class="nx-metric">
                <div class="nx-metric-label">Mean Inference</div>
                <div class="nx-metric-value">{avg_time:.0f}<small>ms</small></div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
            <div class="nx-metric">
                <div class="nx-metric-label">Threshold</div>
                <div class="nx-metric-value">{st.session_state.threshold*100:.0f}<small>%</small></div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="nx-card-title">Recent Studies (this session)</div>', unsafe_allow_html=True)
    if not st.session_state.studies:
        st.markdown("""
            <div class="nx-card" style="text-align:center; padding:3rem 1rem; color:#94a3b8;">
                No studies analyzed yet. Begin by uploading a radiograph in <strong>New Study</strong>.
            </div>
        """, unsafe_allow_html=True)
    else:
        rows_html = []
        for s in reversed(st.session_state.studies[-8:]):
            flagged_html = (
                '<span class="nx-sev nx-sev-high">FLAGGED</span>'
                if s["flagged"] else
                '<span class="nx-sev nx-sev-norm">CLEAR</span>'
            )
            rows_html.append(f"""
                <tr>
                    <td>{s["time"]}</td>
                    <td>{s["filename"][:32]}</td>
                    <td>{s["top_name"]}</td>
                    <td>{s["top_score"]*100:.1f}%</td>
                    <td>{s["elapsed_ms"]:.0f} ms</td>
                    <td>{flagged_html}</td>
                </tr>
            """)
        st.markdown(f"""
            <div class="nx-card" style="padding:0;">
            <table class="nx-table">
                <thead>
                    <tr>
                        <th>Time</th><th>File</th><th>Top Finding</th>
                        <th>Confidence</th><th>Inference</th><th>Status</th>
                    </tr>
                </thead>
                <tbody>{''.join(rows_html)}</tbody>
            </table>
            </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE: NEW STUDY
# ============================================================
elif page == "New Study":
    topbar("New Study")

    st.markdown('<div class="nx-h1">New Study</div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-h1-sub">Upload a chest radiograph for AI-assisted analysis. All outputs require radiologist verification.</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.15], gap="large")

    with col_left:
        uploaded_file = st.file_uploader(
            "Radiograph input (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            filesize_kb = len(file_bytes) / 1024
            input_image = Image.open(io.BytesIO(file_bytes)).convert("L")
            st.image(input_image, use_container_width=True)

            st.markdown(f"""
                <div class="nx-meta-row" style="margin-top:1rem;">
                    <span class="nx-chip">FILE: {uploaded_file.name[:24]}</span>
                    <span class="nx-chip">{filesize_kb:.0f} KB</span>
                    <span class="nx-chip">{input_image.size[0]}×{input_image.size[1]}</span>
                </div>
            """, unsafe_allow_html=True)

            run_btn = st.button("Run AI Analysis", type="primary", use_container_width=True)

            if run_btn:
                with st.spinner("Analyzing radiograph..."):
                    result = run_inference(input_image)
                    result["flagged"] = result["top_score"] >= st.session_state.threshold
                    result["time"] = datetime.now().strftime("%H:%M:%S")
                    result["filename"] = uploaded_file.name
                    result["filesize"] = filesize_kb
                    st.session_state.active_result = result
                    st.session_state.active_image = input_image
                    st.session_state.active_filename = uploaded_file.name
                    st.session_state.active_filesize = filesize_kb
                    st.session_state.studies.append({
                        "time": result["time"],
                        "filename": uploaded_file.name,
                        "top_name": result["top_name"],
                        "top_score": result["top_score"],
                        "elapsed_ms": result["elapsed_ms"],
                        "flagged": result["flagged"],
                    })

    with col_right:
        result = st.session_state.active_result

        if result is None:
            st.markdown("""
                <div class="nx-card" style="text-align:center; padding:3.5rem 1.5rem;">
                    <div style="font-size:0.72rem; color:#94a3b8; letter-spacing:0.14em;
                                text-transform:uppercase; font-family:'IBM Plex Mono', monospace;">
                        Awaiting Input
                    </div>
                    <div style="margin-top:1rem; font-size:0.92rem; color:#64748b;">
                        Upload a radiograph and run analysis to view diagnostic probabilities.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            flagged = result["flagged"]
            banner_class = "nx-banner-flag" if flagged else "nx-banner-clear"
            banner_icon = "!" if flagged else "✓"
            banner_text = (
                f"Pathologies flagged above {st.session_state.threshold*100:.0f}% clinical threshold"
                if flagged else
                f"No pathologies flagged above {st.session_state.threshold*100:.0f}% clinical threshold"
            )

            st.markdown(f"""
                <div class="nx-banner {banner_class}">
                    <div class="nx-banner-icon">{banner_icon}</div>
                    <div>{banner_text}</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class="nx-meta-row">
                    <span class="nx-chip nx-chip-strong">MODEL: DENSENET-121</span>
                    <span class="nx-chip">INFERENCE: {result['elapsed_ms']:.0f} ms</span>
                    <span class="nx-chip">LABELS: 18</span>
                </div>
            """, unsafe_allow_html=True)
            rows_html = []
            for name, score in result["pairs"]:
                sev = severity_of(score, st.session_state.threshold)
                sev_label = severity_label(sev)
                bar_cls = f"nx-bar-{sev}"
                sev_cls = f"nx-sev-{sev}"
                width = int(score * 100)
                rows_html.append(
                    f'<div class="nx-row">'
                    f'<div class="nx-row-label">{name}</div>'
                    f'<div class="nx-row-bar"><div class="nx-row-bar-fill {bar_cls}" style="width:{width}%"></div></div>'
                    f'<div class="nx-row-value">{score*100:.1f}%</div>'
                    f'<div class="nx-row-sev"><span class="nx-sev {sev_cls}">{sev_label}</span></div>'
                    f'</div>'
                )

            st.markdown(
                f'<div class="nx-card">'
                f'<div class="nx-card-title">Diagnostic Probabilities</div>'
                f'{"".join(rows_html)}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Attention map — full width below
    if st.session_state.active_result is not None and st.session_state.active_result.get("cam") is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class="nx-card-title">AI Reasoning — Attention Map (Grad-CAM)</div>
        """, unsafe_allow_html=True)
        st.caption(
            f"Region of interest for top prediction: **{st.session_state.active_result['top_name']}** "
            f"({st.session_state.active_result['top_score']*100:.1f}%)"
        )
        st.image(st.session_state.active_result["cam"], use_container_width=False, width=520)

    st.markdown("""
        <div class="nx-card" style="background:#f8fafc; border-left:3px solid #64748b; margin-top:2rem;">
            <strong style="color:#0a1628;">Clinical Disclaimer.</strong>
            This output is generated by a research prototype and is <strong>not a medical diagnosis</strong>.
            It is intended for investigational and educational use only. All findings must be reviewed
            by a qualified radiologist or clinician before any clinical decision.
        </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE: WORKLIST
# ============================================================
elif page == "Worklist":
    topbar("Worklist")

    st.markdown('<div class="nx-h1">Worklist</div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-h1-sub">Studies analyzed this session. Persistent worklist requires authenticated backend.</div>', unsafe_allow_html=True)

    if not st.session_state.studies:
        st.markdown("""
            <div class="nx-card" style="text-align:center; padding:3rem 1rem; color:#94a3b8;">
                Worklist is empty. Studies will appear here after analysis in <strong>New Study</strong>.
            </div>
        """, unsafe_allow_html=True)
    else:
        rows_html = []
        for i, s in enumerate(st.session_state.studies, 1):
            status = (
                '<span class="nx-sev nx-sev-high">FLAGGED</span>'
                if s["flagged"] else
                '<span class="nx-sev nx-sev-norm">CLEAR</span>'
            )
            rows_html.append(f"""
                <tr>
                    <td>{i:03d}</td>
                    <td>{s["time"]}</td>
                    <td>{s["filename"][:40]}</td>
                    <td>{s["top_name"]}</td>
                    <td>{s["top_score"]*100:.1f}%</td>
                    <td>{s["elapsed_ms"]:.0f} ms</td>
                    <td>{status}</td>
                </tr>
            """)
        st.markdown(f"""
            <div class="nx-card" style="padding:0;">
            <table class="nx-table">
                <thead>
                    <tr>
                        <th>#</th><th>Time</th><th>Filename</th>
                        <th>Top Finding</th><th>Confidence</th>
                        <th>Inference</th><th>Status</th>
                    </tr>
                </thead>
                <tbody>{''.join(rows_html)}</tbody>
            </table>
            </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE: REPORTS
# ============================================================
elif page == "Reports":
    topbar("Reports")

    st.markdown('<div class="nx-h1">Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-h1-sub">Export session data for review or audit.</div>', unsafe_allow_html=True)

    if not st.session_state.studies:
        st.markdown("""
            <div class="nx-card" style="text-align:center; padding:3rem 1rem; color:#94a3b8;">
                No studies to report. Analyze a radiograph first.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="nx-card-title">Session Summary</div>', unsafe_allow_html=True)
        total = len(st.session_state.studies)
        flagged = sum(1 for s in st.session_state.studies if s["flagged"])
        clear = total - flagged

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="nx-metric"><div class="nx-metric-label">Total Studies</div><div class="nx-metric-value">{total}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="nx-metric"><div class="nx-metric-label">Flagged</div><div class="nx-metric-value" style="color:#dc2626;">{flagged}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="nx-metric"><div class="nx-metric-label">Clear</div><div class="nx-metric-value" style="color:#059669;">{clear}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="nx-card-title">Export</div>', unsafe_allow_html=True)

        # Build CSV
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["timestamp", "filename", "top_finding", "confidence", "inference_ms", "status"])
        for s in st.session_state.studies:
            writer.writerow([
                s["time"], s["filename"], s["top_name"],
                f"{s['top_score']:.4f}", f"{s['elapsed_ms']:.1f}",
                "FLAGGED" if s["flagged"] else "CLEAR",
            ])
        csv_data = buf.getvalue()

        st.download_button(
            label="Download Session CSV",
            data=csv_data,
            file_name=f"navantix_session_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=False,
        )


# ============================================================
# PAGE: SETTINGS
# ============================================================
elif page == "Settings":
    topbar("Settings")

    st.markdown('<div class="nx-h1">Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-h1-sub">Adjust analysis parameters. Changes apply to subsequent studies.</div>', unsafe_allow_html=True)

    st.markdown('<div class="nx-card-title">Clinical Threshold</div>', unsafe_allow_html=True)
    new_threshold = st.slider(
        "Flag pathologies above this probability",
        min_value=0.1, max_value=0.9,
        value=st.session_state.threshold,
        step=0.05,
        format="%.0f%%" if False else "%.2f",
    )
    st.session_state.threshold = new_threshold
    st.caption(f"Current threshold: **{new_threshold*100:.0f}%**. Findings above this are marked FLAGGED.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="nx-card-title">Model Information</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="nx-card">
            <table class="nx-table">
                <tbody>
                    <tr><td style="color:#64748b;">Architecture</td><td>DenseNet-121</td></tr>
                    <tr><td style="color:#64748b;">Weights</td><td>densenet121-res224-all (TorchXRayVision)</td></tr>
                    <tr><td style="color:#64748b;">Training Data</td><td>NIH ChestX-ray14 + additional public datasets</td></tr>
                    <tr><td style="color:#64748b;">Input Resolution</td><td>224 × 224</td></tr>
                    <tr><td style="color:#64748b;">Output Labels</td><td>18 chest conditions</td></tr>
                    <tr><td style="color:#64748b;">Explainability</td><td>Grad-CAM (features.norm5)</td></tr>
                </tbody>
            </table>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="nx-card-title">Session</div>', unsafe_allow_html=True)
    if st.button("Clear Session Data", use_container_width=False):
        st.session_state.studies = []
        st.session_state.active_result = None
        st.session_state.active_image = None
        st.success("Session cleared.")


# ============================================================
# FOOTER
# ============================================================
st.markdown(f"""
    <div class="nx-footer">
        <div>NAVANTIX PULMO v1.0 · RESEARCH PROTOTYPE · NOT FOR CLINICAL USE</div>
        <div>BUILT IN GHANA · {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
    </div>
""", unsafe_allow_html=True)
