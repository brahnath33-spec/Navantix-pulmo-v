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
# STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'IBM Plex Sans', -apple-system, sans-serif;
        color: #1a1f2e;
        background: #f5f7fa;
    }
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="stDecoration"], [data-testid="stToolbar"] {display: none;}

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0a1628;
        border-right: 1px solid #1a2740;
    }
    [data-testid="stSidebar"] * { color: #cbd5e1; }
    [data-testid="stSidebar"] .stRadio > label { display: none; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        padding: 0.7rem 1rem;
        border-radius: 4px;
        margin: 0.15rem 0;
        cursor: pointer;
        font-size: 0.92rem;
        color: #94a3b8;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background: #14213a;
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
        background: #1a2f52;
        color: #ffffff;
        font-weight: 500;
    }
    [data-testid="stSidebar"] input[type="radio"] { display: none; }

    .main .block-container {
        max-width: 1200px;
        padding: 1.5rem 2rem 3rem 2rem;
    }

    .nx-brand-block {
        padding: 1.4rem 1rem 1.2rem 1rem;
        border-bottom: 1px solid #1a2740;
        margin-bottom: 0.8rem;
    }
    .nx-brand-mark {
        font-size: 1.5rem; font-weight: 700; color: #ffffff;
        letter-spacing: 0.02em; line-height: 1;
    }
    .nx-brand-mark span { color: #3b82f6; }
    .nx-brand-sub {
        font-size: 0.6rem; color: #64748b;
        letter-spacing: 0.18em; text-transform: uppercase;
        margin-top: 0.4rem; font-family: 'IBM Plex Mono', monospace;
    }

    .nx-topbar {
        display: flex; justify-content: space-between; align-items: center;
        padding-bottom: 1rem; border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    .nx-breadcrumb { font-size: 0.85rem; color: #64748b; }
    .nx-breadcrumb strong { color: #0a1628; font-weight: 600; }
    .nx-status {
        display: inline-flex; align-items: center; gap: 0.5rem;
        padding: 0.35rem 0.8rem; background: #ecfdf5;
        border: 1px solid #a7f3d0; border-radius: 999px;
        font-size: 0.72rem; color: #047857; font-weight: 500;
    }
    .nx-status-dot {
        width: 6px; height: 6px; border-radius: 50%; background: #10b981;
    }

    .nx-h1 {
        font-size: 1.5rem; font-weight: 600; color: #0a1628;
        margin: 0 0 0.25rem 0; letter-spacing: -0.01em;
    }
    .nx-h1-sub {
        font-size: 0.88rem; color: #64748b; margin-bottom: 1.75rem;
    }

    .nx-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
    }
    .nx-card-title {
        font-size: 0.7rem; font-weight: 600;
        letter-spacing: 0.12em; text-transform: uppercase;
        color: #64748b; margin-bottom: 0.9rem;
    }

    .nx-sev {
        display: inline-block; font-size: 0.66rem; font-weight: 700;
        letter-spacing: 0.08em; padding: 0.18rem 0.5rem;
        border-radius: 3px; text-transform: uppercase;
        font-family: 'IBM Plex Mono', monospace;
    }
    .nx-sev-high { background: #fee2e2; color: #991b1b; }
    .nx-sev-mod  { background: #fef3c7; color: #92400e; }
    .nx-sev-low  { background: #e2e8f0; color: #475569; }
    .nx-sev-norm { background: #d1fae5; color: #065f46; }

    .nx-row {
        display: grid;
        grid-template-columns: 200px 1fr 75px 90px;
        align-items: center; gap: 0.6rem;
        padding: 0.65rem 0;
        border-bottom: 1px solid #eef2f7;
        font-size: 0.88rem;
    }
    .nx-row:last-child { border-bottom: none; }
    .nx-row-label { color: #1e293b; }
    .nx-row-bar {
        height: 6px; background: #eef2f7;
        border-radius: 3px; overflow: hidden;
    }
    .nx-row-bar-fill { height: 100%; border-radius: 3px; }
    .nx-bar-high { background: linear-gradient(90deg, #dc2626, #ef4444); }
    .nx-bar-mod  { background: linear-gradient(90deg, #d97706, #f59e0b); }
    .nx-bar-low  { background: linear-gradient(90deg, #64748b, #94a3b8); }
    .nx-row-value {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
        font-weight: 500; text-align: right; color: #0a1628;
    }
    .nx-row-sev { text-align: right; }

    .nx-banner {
        padding: 1rem 1.3rem; border-radius: 8px;
        display: flex; align-items: center; gap: 1rem;
        margin-bottom: 1.25rem; font-size: 0.92rem; font-weight: 500;
    }
    .nx-banner-clear { background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46; }
    .nx-banner-flag  { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }
    .nx-banner-icon { font-size: 1.2rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }

    .nx-meta-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.8rem; }
    .nx-chip {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
        padding: 0.28rem 0.65rem; background: #eef2f7;
        border-radius: 3px; color: #475569; letter-spacing: 0.02em;
    }
    .nx-chip-strong { background: #0a1628; color: #ffffff; }

    .stButton > button {
        background: #0a1628; color: #ffffff; border: none;
        border-radius: 5px; padding: 0.6rem 1.3rem;
        font-weight: 500; font-size: 0.86rem;
        font-family: 'IBM Plex Sans', sans-serif;
        letter-spacing: 0.02em; width: 100%;
    }
    .stButton > button:hover { background: #1e3a8a; color: #ffffff; }
    .stButton > button[kind="primary"] { background: #2563eb; }
    .stButton > button[kind="primary"]:hover { background: #1d4ed8; }

    [data-testid="stFileUploader"] {
        border: 1.5px dashed #cbd5e1; border-radius: 8px;
        background: #ffffff; padding: 0.9rem;
    }
    [data-testid="stFileUploader"]:hover { border-color: #3b82f6; background: #f8fafc; }

    .nx-footer {
        border-top: 1px solid #e2e8f0; margin-top: 2.5rem;
        padding-top: 1rem; font-size: 0.68rem; color: #94a3b8;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.04em;
        display: flex; justify-content: space-between;
    }

    .nx-metric {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 1.1rem;
    }
    .nx-metric-label {
        font-size: 0.68rem; color: #64748b; text-transform: uppercase;
        letter-spacing: 0.1em; font-weight: 600; margin-bottom: 0.4rem;
    }
    .nx-metric-value {
        font-size: 1.5rem; font-weight: 600; color: #0a1628;
        font-family: 'IBM Plex Mono', monospace;
    }
    .nx-metric-value small {
        font-size: 0.72rem; color: #94a3b8;
        margin-left: 0.25rem; font-weight: 400;
    }

    .nx-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    .nx-table thead th {
        text-align: left; padding: 0.65rem 0.85rem;
        font-size: 0.68rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.1em;
        color: #64748b; border-bottom: 1px solid #e2e8f0;
        background: #f8fafc;
    }
    .nx-table tbody td {
        padding: 0.7rem 0.85rem;
        border-bottom: 1px solid #eef2f7;
        color: #1e293b;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
    }
    .nx-table tbody tr:hover { background: #f8fafc; }

    .nx-empty {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 3rem 1.5rem;
        text-align: center; color: #94a3b8; font-size: 0.9rem;
    }
    .nx-empty-strong {
        color: #0a1628; font-weight: 600; font-size: 0.95rem;
        margin-bottom: 0.4rem;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.1em; text-transform: uppercase;
    }

    [data-testid="stSlider"] label {
        font-size: 0.8rem; color: #475569; font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    m = xrv.models.DenseNet(weights="densenet121-res224-all")
    m.eval()
    return m

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
    mx = float(img_array.max()); mn = float(img_array.min())
    if mx <= 1.0:
        img_array = img_array * 255.0
    elif mx > 255.0:
        img_array = (img_array - mn) / (mx - mn) * 255.0
    img = xrv.datasets.normalize(img_array, 255)
    if len(img.shape) > 2:
        img = img[:, :, 0] if img.shape[2] == 1 else img.mean(axis=2)
    img = img[None, ...]
    t = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    return torch.from_numpy(t(img))


def run_inference(input_image):
    t0 = time.time()
    arr = np.array(input_image.convert("L"))
    tensor = prepare_xray(arr)
    with torch.no_grad():
        outputs = model(tensor[None, ...])
    scores = outputs[0].numpy()
    pairs = sorted(zip(model.pathologies, scores), key=lambda x: -x[1])
    cam = None
    try:
        top_idx = model.pathologies.index(pairs[0][0])
        gc = GradCAM(model=model, target_layers=[model.features.norm5])
        gcam = gc(input_tensor=tensor[None, ...],
                  targets=[ClassifierOutputTarget(top_idx)])[0]
        disp = np.array(input_image.convert("L").resize((224, 224))) / 255.0
        disp = np.stack([disp] * 3, axis=-1).astype(np.float32)
        cam = show_cam_on_image(disp, gcam, use_rgb=True)
    except Exception:
        cam = None
    return {
        "pairs": pairs, "cam": cam,
        "elapsed_ms": (time.time() - t0) * 1000.0,
        "top_name": pairs[0][0], "top_score": float(pairs[0][1]),
    }


def severity_of(score, threshold):
    if score >= threshold + 0.1: return "high"
    if score >= threshold: return "mod"
    return "low"


def severity_label(sev):
    return {"high": "HIGH", "mod": "MODERATE", "low": "LOW"}[sev]


# ============================================================
# SIDEBAR
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
        key="main_nav",
    )

    st.markdown("""
        <div style="position:fixed; bottom:1rem; left:1rem; right:1rem;
                    padding-top:1rem; border-top:1px solid #1a2740;
                    font-size:0.66rem; color:#64748b;
                    font-family:'IBM Plex Mono', monospace;
                    letter-spacing:0.04em;">
            ADMIN@NAVANTIXPULMO.LOCAL<br>
            <span style="color:#3b82f6;">●</span> CONNECTED
        </div>
    """, unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "studies": [],
    "threshold": 0.5,
    "active_result": None,
    "active_image": None,
    "active_filename": None,
    "active_filesize": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_study():
    st.session_state.active_result = None
    st.session_state.active_image = None
    st.session_state.active_filename = None
    st.session_state.active_filesize = None


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
    for col, label, val in [
        (c1, "Studies Analyzed", f"{n_studies}"),
        (c2, "Flagged Studies", f"{n_flagged}"),
        (c3, "Mean Inference", f"{avg_time:.0f}<small>ms</small>"),
        (c4, "Threshold", f"{st.session_state.threshold*100:.0f}<small>%</small>"),
    ]:
        with col:
            st.markdown(
                f'<div class="nx-metric">'
                f'<div class="nx-metric-label">{label}</div>'
                f'<div class="nx-metric-value">{val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="nx-card-title">Recent Studies</div>', unsafe_allow_html=True)

    if not st.session_state.studies:
        st.markdown(
            '<div class="nx-empty">'
            '<div class="nx-empty-strong">No studies yet</div>'
            'Begin by uploading a radiograph on the <strong>New Study</strong> page.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        rows = []
        for s in reversed(st.session_state.studies[-8:]):
            st_html = ('<span class="nx-sev nx-sev-high">FLAGGED</span>'
                       if s["flagged"] else
                       '<span class="nx-sev nx-sev-norm">CLEAR</span>')
            rows.append(
                f'<tr><td>{s["time"]}</td><td>{s["filename"][:30]}</td>'
                f'<td>{s["top_name"]}</td><td>{s["top_score"]*100:.1f}%</td>'
                f'<td>{s["elapsed_ms"]:.0f} ms</td><td>{st_html}</td></tr>'
            )
        st.markdown(
            f'<div class="nx-card" style="padding:0;">'
            f'<table class="nx-table"><thead><tr>'
            f'<th>Time</th><th>File</th><th>Top Finding</th>'
            f'<th>Confidence</th><th>Inference</th><th>Status</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE: NEW STUDY
# ============================================================
elif page == "New Study":
    topbar("New Study")
    st.markdown('<div class="nx-h1">New Study</div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-h1-sub">Upload a chest radiograph for AI-assisted analysis.</div>', unsafe_allow_html=True)

    # -------- Upload row --------
    up_col, btn_col = st.columns([3, 1])
    with up_col:
        uploaded_file = st.file_uploader(
            "Radiograph input",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
    with btn_col:
        st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
        run_btn = st.button("Run AI Analysis", type="primary", use_container_width=True)

    # -------- If a file is uploaded --------
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        filesize_kb = len(file_bytes) / 1024
        input_image = Image.open(io.BytesIO(file_bytes)).convert("L")

        if run_btn:
            with st.spinner("Analyzing radiograph..."):
                result = run_inference(input_image)
                result["flagged"] = result["top_score"] >= st.session_state.threshold
                result["time"] = datetime.now().strftime("%H:%M:%S")
                result["filename"] = uploaded_file.name
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

        result = st.session_state.active_result

        # Decision banner
        if result is not None:
            flagged = result["flagged"]
            cls = "nx-banner-flag" if flagged else "nx-banner-clear"
            icon = "!" if flagged else "✓"
            text = (f"Pathologies flagged above {st.session_state.threshold*100:.0f}% threshold"
                    if flagged else
                    f"No pathologies flagged above {st.session_state.threshold*100:.0f}% threshold")
            st.markdown(
                f'<div class="nx-banner {cls}">'
                f'<div class="nx-banner-icon">{icon}</div>'
                f'<div>{text}</div></div>',
                unsafe_allow_html=True,
            )

        # Two-column: image | probabilities
        col_img, col_res = st.columns([1, 1.6], gap="large")

        with col_img:
            st.markdown('<div class="nx-card-title">Radiograph</div>', unsafe_allow_html=True)
            st.image(input_image, use_container_width=True)
            st.markdown(
                f'<div class="nx-meta-row" style="margin-top:0.8rem;">'
                f'<span class="nx-chip">FILE: {uploaded_file.name[:20]}</span>'
                f'<span class="nx-chip">{filesize_kb:.0f} KB</span>'
                f'<span class="nx-chip">{input_image.size[0]}×{input_image.size[1]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col_res:
            if result is None:
                st.markdown(
                    '<div class="nx-empty">'
                    '<div class="nx-empty-strong">Awaiting Analysis</div>'
                    'Click <strong>Run AI Analysis</strong> to generate diagnostic probabilities.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="nx-card-title">Diagnostic Probabilities</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="nx-meta-row">'
                    f'<span class="nx-chip nx-chip-strong">DENSENET-121</span>'
                    f'<span class="nx-chip">{result["elapsed_ms"]:.0f} ms</span>'
                    f'<span class="nx-chip">18 LABELS</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                rows = []
                for name, score in result["pairs"]:
                    sev = severity_of(score, st.session_state.threshold)
                    rows.append(
                        f'<div class="nx-row">'
                        f'<div class="nx-row-label">{name}</div>'
                        f'<div class="nx-row-bar">'
                        f'<div class="nx-row-bar-fill nx-bar-{sev}" style="width:{int(score*100)}%"></div>'
                        f'</div>'
                        f'<div class="nx-row-value">{score*100:.1f}%</div>'
                        f'<div class="nx-row-sev"><span class="nx-sev nx-sev-{sev}">{severity_label(sev)}</span></div>'
                        f'</div>'
                    )
                st.markdown(
                    f'<div class="nx-card" style="padding:0.75rem 1.25rem;">'
                    f'{"".join(rows)}</div>',
                    unsafe_allow_html=True,
                )

        # Attention map
        if result is not None and result.get("cam") is not None:
            with st.expander("AI Reasoning — Attention Map (Grad-CAM)", expanded=False):
                st.caption(
                    f"Region of interest for top prediction: **{result['top_name']}** "
                    f"({result['top_score']*100:.1f}%)"
                )
                st.image(result["cam"], width=420)

        # Action buttons
        st.markdown("<br>", unsafe_allow_html=True)
        ac1, ac2, _ = st.columns([1, 1, 2])
        with ac1:
            if st.button("Clear / New Study", use_container_width=True):
                reset_study()
                st.rerun()
        with ac2:
            if result is not None:
                st.caption(f"Analysis saved to Worklist at {result['time']}")

    else:
        st.markdown(
            '<div class="nx-empty" style="margin-top:1rem;">'
            '<div class="nx-empty-strong">Awaiting Radiograph</div>'
            'Upload a chest X-ray (JPG, JPEG, or PNG) to begin analysis.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="nx-card" style="background:#f8fafc; border-left:3px solid #64748b; margin-top:1.5rem;">'
        '<strong style="color:#0a1628;">Clinical Disclaimer.</strong> '
        'This output is generated by a research prototype and is <strong>not a medical diagnosis</strong>. '
        'All findings must be reviewed by a qualified radiologist before any clinical decision.'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE: WORKLIST
# ============================================================
elif page == "Worklist":
    topbar("Worklist")
    st.markdown('<div class="nx-h1">Worklist</div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-h1-sub">All studies analyzed during this session.</div>', unsafe_allow_html=True)

    if not st.session_state.studies:
        st.markdown(
            '<div class="nx-empty">'
            '<div class="nx-empty-strong">Worklist is empty</div>'
            'Studies will appear here after analysis on the <strong>New Study</strong> page.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        rows = []
        for i, s in enumerate(st.session_state.studies, 1):
            st_html = ('<span class="nx-sev nx-sev-high">FLAGGED</span>'
                       if s["flagged"] else
                       '<span class="nx-sev nx-sev-norm">CLEAR</span>')
            rows.append(
                f'<tr><td>{i:03d}</td><td>{s["time"]}</td>'
                f'<td>{s["filename"][:40]}</td><td>{s["top_name"]}</td>'
                f'<td>{s["top_score"]*100:.1f}%</td>'
                f'<td>{s["elapsed_ms"]:.0f} ms</td><td>{st_html}</td></tr>'
            )
        st.markdown(
            f'<div class="nx-card" style="padding:0;">'
            f'<table class="nx-table"><thead><tr>'
            f'<th>#</th><th>Time</th><th>Filename</th>'
            f'<th>Top Finding</th><th>Confidence</th>'
            f'<th>Inference</th><th>Status</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE: REPORTS
# ============================================================
elif page == "Reports":
    topbar("Reports")
    st.markdown('<div class="nx-h1">Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-h1-sub">Session summary and export.</div>', unsafe_allow_html=True)

    if not st.session_state.studies:
        st.markdown(
            '<div class="nx-empty">'
            '<div class="nx-empty-strong">No data to report</div>'
            'Analyze at least one radiograph to generate a report.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        total = len(st.session_state.studies)
        flagged = sum(1 for s in st.session_state.studies if s["flagged"])
        clear = total - flagged

        c1, c2, c3 = st.columns(3)
        for col, label, val, color in [
            (c1, "Total Studies", f"{total}", "#0a1628"),
            (c2, "Flagged", f"{flagged}", "#dc2626"),
            (c3, "Clear", f"{clear}", "#059669"),
        ]:
            with col:
                st.markdown(
                    f'<div class="nx-metric">'
                    f'<div class="nx-metric-label">{label}</div>'
                    f'<div class="nx-metric-value" style="color:{color};">{val}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="nx-card-title">Export Session</div>', unsafe_allow_html=True)

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["timestamp", "filename", "top_finding", "confidence",
                    "inference_ms", "status"])
        for s in st.session_state.studies:
            w.writerow([s["time"], s["filename"], s["top_name"],
                        f"{s['top_score']:.4f}", f"{s['elapsed_ms']:.1f}",
                        "FLAGGED" if s["flagged"] else "CLEAR"])

        st.download_button(
            label="Download Session CSV",
            data=buf.getvalue(),
            file_name=f"navantix_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )


# ============================================================
# PAGE: SETTINGS
# ============================================================
elif page == "Settings":
    topbar("Settings")
    st.markdown('<div class="nx-h1">Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-h1-sub">Analysis parameters and session control.</div>', unsafe_allow_html=True)

    st.markdown('<div class="nx-card-title">Clinical Threshold</div>', unsafe_allow_html=True)
    new_threshold = st.slider(
        "Flag pathologies above this probability",
        min_value=0.1, max_value=0.9,
        value=st.session_state.threshold,
        step=0.05,
        format="%.2f",
    )
    st.session_state.threshold = new_threshold
    st.caption(f"Current: **{new_threshold*100:.0f}%**. Findings above are marked FLAGGED.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="nx-card-title">Model Information</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="nx-card"><table class="nx-table"><tbody>'
        '<tr><td style="color:#64748b;">Architecture</td><td>DenseNet-121</td></tr>'
        '<tr><td style="color:#64748b;">Weights</td><td>densenet121-res224-all</td></tr>'
        '<tr><td style="color:#64748b;">Input</td><td>224 × 224 grayscale</td></tr>'
        '<tr><td style="color:#64748b;">Labels</td><td>18 chest conditions</td></tr>'
        '<tr><td style="color:#64748b;">Explainability</td><td>Grad-CAM</td></tr>'
        '</tbody></table></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="nx-card-title">Session</div>', unsafe_allow_html=True)
    if st.button("Clear Session Data", use_container_width=False):
        st.session_state.studies = []
        reset_study()
        st.success("Session cleared.")


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    f'<div class="nx-footer">'
    f'<div>NAVANTIX PULMO v1.0 · RESEARCH PROTOTYPE · NOT FOR CLINICAL USE</div>'
    f'<div>BUILT IN GHANA · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>'
    f'</div>',
    unsafe_allow_html=True,
)
