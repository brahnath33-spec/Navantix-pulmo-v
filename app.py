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

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Navantix Pulmo",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# PROFESSIONAL STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', -apple-system, sans-serif;
        color: #1a1f2e;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .nx-header {
        border-bottom: 1px solid #e2e6ed;
        padding-bottom: 1.25rem;
        margin-bottom: 2rem;
    }
    .nx-brand {
        font-size: 1.35rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #0a1628;
        margin: 0;
    }
    .nx-subtitle {
        font-size: 0.82rem;
        font-weight: 400;
        color: #5a6577;
        letter-spacing: 0.02em;
        margin-top: 0.35rem;
    }
    .nx-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #8891a3;
        letter-spacing: 0.03em;
        margin-top: 0.6rem;
    }

    /* Section headers */
    .nx-section {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #5a6577;
        margin-bottom: 0.85rem;
        margin-top: 0.25rem;
    }

    /* Results table */
    .nx-result-row {
        display: flex;
        align-items: center;
        padding: 0.55rem 0;
        border-bottom: 1px solid #eef1f6;
        font-size: 0.92rem;
    }
    .nx-result-label {
        flex: 1;
        color: #2a3244;
        font-weight: 400;
    }
    .nx-result-bar-wrap {
        width: 180px;
        height: 5px;
        background: #eef1f6;
        border-radius: 3px;
        margin: 0 1rem;
        overflow: hidden;
    }
    .nx-result-bar {
        height: 100%;
        border-radius: 3px;
    }
    .nx-result-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        font-weight: 500;
        width: 55px;
        text-align: right;
        color: #2a3244;
    }
    .nx-result-flag {
        display: inline-block;
        width: 4px;
        height: 4px;
        border-radius: 50%;
        margin-right: 0.6rem;
    }
    .nx-flag-high { background: #c1272d; }
    .nx-flag-mid  { background: #d97706; }
    .nx-flag-low  { background: #cbd2dc; }

    .nx-bar-high { background: #c1272d; }
    .nx-bar-mid  { background: #d97706; }
    .nx-bar-low  { background: #8891a3; }

    /* Disclaimer */
    .nx-disclaimer {
        background: #f7f8fa;
        border-left: 3px solid #8891a3;
        padding: 0.9rem 1.1rem;
        font-size: 0.78rem;
        color: #5a6577;
        line-height: 1.55;
        margin-top: 1.5rem;
        border-radius: 2px;
    }
    .nx-disclaimer strong { color: #2a3244; }

    /* Footer */
    .nx-footer {
        border-top: 1px solid #e2e6ed;
        margin-top: 3rem;
        padding-top: 1.25rem;
        font-size: 0.72rem;
        color: #8891a3;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.02em;
    }

    /* File uploader cleanup */
    [data-testid="stFileUploader"] {
        border: 1px dashed #cbd2dc;
        border-radius: 4px;
        background: #fafbfc;
    }

    /* Hide Streamlit default menu */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model

model = load_model()


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


def flag_class(score):
    if score >= 0.5:
        return "high"
    if score >= 0.3:
        return "mid"
    return "low"


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="nx-header">
    <div class="nx-brand">Navantix Pulmo</div>
    <div class="nx-subtitle">AI-Assisted Chest Radiograph Analysis</div>
    <div class="nx-meta">MODEL: DENSENET-121 · WEIGHTS: densenet121-res224-all · INPUT: 224×224 · LABELS: 18</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# INPUT
# ============================================================
st.markdown('<div class="nx-section">Radiograph Input</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a chest radiograph (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)


# ============================================================
# INFERENCE
# ============================================================
if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("L")
    img_array = np.array(input_image)
    img_tensor = prepare_xray(img_array)

    with st.spinner("Analyzing radiograph..."):
        with torch.no_grad():
            outputs = model(img_tensor[None, ...])
        scores = outputs[0].numpy()

    pairs = sorted(zip(model.pathologies, scores), key=lambda x: -x[1])

    # ---------------- Results layout ----------------
    col_img, col_report = st.columns([1, 1.2], gap="large")

    with col_img:
        st.markdown('<div class="nx-section">Radiograph</div>', unsafe_allow_html=True)
        st.image(input_image, use_container_width=True)

    with col_report:
        st.markdown('<div class="nx-section">Diagnostic Probabilities</div>', unsafe_allow_html=True)

        rows_html = []
        for name, score in pairs:
            cls = flag_class(score)
            bar_width = int(score * 100)
            rows_html.append(f"""
            <div class="nx-result-row">
                <span class="nx-result-flag nx-flag-{cls}"></span>
                <span class="nx-result-label">{name}</span>
                <span class="nx-result-bar-wrap">
                    <span class="nx-result-bar nx-bar-{cls}" style="width:{bar_width}%"></span>
                </span>
                <span class="nx-result-value">{score*100:.1f}%</span>
            </div>
            """)

        st.markdown("".join(rows_html), unsafe_allow_html=True)

    # ---------------- Attention map ----------------
    st.markdown('<div class="nx-section" style="margin-top:2rem;">Attention Map</div>', unsafe_allow_html=True)
    st.caption(f"Region of interest for: {pairs[0][0]}")

    try:
        top_name, _ = pairs[0]
        top_idx = model.pathologies.index(top_name)
        target_layer = [model.features.norm5]
        cam = GradCAM(model=model, target_layers=target_layer)
        grayscale_cam = cam(
            input_tensor=img_tensor[None, ...],
            targets=[ClassifierOutputTarget(top_idx)],
        )[0]

        display_img = np.array(input_image.resize((224, 224))) / 255.0
        display_img = np.stack([display_img] * 3, axis=-1).astype(np.float32)
        overlay = show_cam_on_image(display_img, grayscale_cam, use_rgb=True)
        st.image(overlay, use_container_width=True)
    except Exception as e:
        st.warning(f"Attention map unavailable: {e}")

    # ---------------- Disclaimer ----------------
    st.markdown("""
    <div class="nx-disclaimer">
        <strong>Clinical Disclaimer.</strong> This output is generated by a research
        prototype and is <strong>not a medical diagnosis</strong>. It is intended for
        investigational and educational use only. All findings must be reviewed by a
        qualified radiologist or clinician before any clinical decision. Do not use
        for primary diagnostic interpretation.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="nx-footer">
    NAVANTIX PULMO v0.1 · RESEARCH PROTOTYPE · NOT FOR CLINICAL USE<br>
    BUILT IN GHANA FOR AFRICAN RADIOLOGISTS
</div>
""", unsafe_allow_html=True)
