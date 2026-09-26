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

# --- Page Config ---
st.set_page_config(page_title="Navantix Pulmo", page_icon="🫁", layout="centered")

# --- Model Loading (Cached for performance) ---
@st.cache_resource
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

# --- UI ---
st.title("🫁 Navantix Pulmo")
st.markdown("""
**AI-assisted chest X-ray analysis.**  
*Research prototype — NOT for clinical use.*  
*Built in Ghana for African radiologists.*
""")

uploaded_file = st.file_uploader("Choose a chest X-ray...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Uploaded X-ray", use_container_width=True)

    # Process
    with st.spinner("Analyzing..."):
        input_image = Image.open(uploaded_file).convert("L")
        img_array = np.array(input_image)
        img_tensor = prepare_xray(img_array)

        with torch.no_grad():
            outputs = model(img_tensor[None, ...])
        scores = outputs[0].numpy()

        # Report
        pairs = sorted(zip(model.pathologies, scores), key=lambda x: -x[1])
        report_lines = ["**NAVANTIX PULMO — AI Analysis**", "---", ""]
        for name, score in pairs[:8]:
            flag = "⚠️" if score >= 0.5 else "  "
            report_lines.append(f"{flag} **{name}**: {score:.2f}")
        report_lines += ["", "---", "⚠️ *Research prototype. Not for clinical use.*"]
        report_text = "\n".join(report_lines)

        with col2:
            st.markdown(report_text)

        # Grad-CAM
        st.image(overlay, caption=f"Attention map for: {top_name}", use_container_width=True)
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
            st.image(overlay, caption=f"Attention map for: {top_name}", use_container_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM visualization skipped: {e}")

st.markdown("---")
st.caption("Built by [ANTWI ABABIO NATHANIEL] | [KOFORIDUA TECHNICAL UNIVERSITY] | [2026]")
