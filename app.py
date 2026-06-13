import os
import json
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

for folder in [
    os.path.join(BASE_DIR, "temp"),
    os.path.join(BASE_DIR, "generated_heatmaps"),
    os.path.join(BASE_DIR, "generated_reports"),
]:
    os.makedirs(folder, exist_ok=True)

from PIL import Image

from utils.model_loader import load_model

from utils.inference import (
    predict,
    prepare_input,
    get_top_predictions,
    get_probability_dict,
)

from utils.gradcam import generate_gradcam
from utils.report_generator import create_pdf_report
from utils.disease_info import DISEASE_INFO

# ==========================================================
# FULL DISEASE NAMES & RISK LEVELS
# ==========================================================

DISEASE_FULL_NAMES = {
    "AK":   "Actinic Keratosis",
    "BCC":  "Basal Cell Carcinoma",
    "BKL":  "Benign Keratosis-like Lesion",
    "DF":   "Dermatofibroma",
    "MEL":  "Melanoma",
    "NV":   "Melanocytic Nevus",
    "SCC":  "Squamous Cell Carcinoma",
    "VASC": "Vascular Lesion",
}

RISK_LEVEL = {
    "AK":   ("⚠️ Moderate Risk", "warning"),
    "BCC":  ("🔴 High Risk",     "error"),
    "BKL":  ("🟢 Low Risk",      "success"),
    "DF":   ("🟢 Low Risk",      "success"),
    "MEL":  ("🔴 High Risk",     "error"),
    "NV":   ("🟢 Low Risk",      "success"),
    "SCC":  ("🔴 High Risk",     "error"),
    "VASC": ("⚠️ Moderate Risk", "warning"),
}

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="DermaVision",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# CSS
# ==========================================================

st.markdown("""
<style>
.main { background-color: #0f172a; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

[data-testid="stMetricValue"] { font-size: 26px; font-weight: 700; }
[data-testid="metric-container"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}

.risk-badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin-top: 0.5rem;
}
.risk-high   { background: #7f1d1d; color: #fca5a5; }
.risk-medium { background: #78350f; color: #fcd34d; }
.risk-low    { background: #14532d; color: #86efac; }

section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}

hr { border-color: #1e293b !important; }
.footer-cap { color: #475569; font-size: 0.78rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD LABELS
# ==========================================================

with open(os.path.join(BASE_DIR, "artifacts", "label_mapping.json"), "r") as f:
    idx2label = json.load(f)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def get_model():
    return load_model(
        os.path.join(BASE_DIR, "models", "swin_tiny_best.pth")
    )

model = get_model()

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    logo_path = os.path.join(BASE_DIR, "assets", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)

    st.markdown("## DermaVision")
    st.markdown("*AI-powered skin lesion analysis*")
    st.markdown("---")

    st.markdown("### 🤖 Model")
    st.markdown("""
| Property | Value |
|---|---|
| Architecture | Swin-T |
| Accuracy | 87.13% |
| Dataset | ISIC 2019 |
| Classes | 8 |
""")

    st.markdown("---")
    st.markdown("### 🏷️ Supported Classes")
    for code, name in DISEASE_FULL_NAMES.items():
        st.markdown(f"- **{code}** — {name}")

    st.markdown("---")
    st.info(
        "**Educational & Research Use Only**\n\n"
        "Not a substitute for professional medical advice. "
        "Always consult a qualified dermatologist."
    )

# ==========================================================
# HEADER
# ==========================================================

st.markdown("# 🩺 DermaVision")
st.markdown(
    "**AI-Powered Skin Lesion Classification** — "
    "Upload a dermoscopic image for instant analysis."
)

# ==========================================================
# METRICS ROW
# ==========================================================

c1, c2, c3, c4 = st.columns(4)
c1.metric("Validation Accuracy", "87.13%")
c2.metric("Training Images",     "25,331")
c3.metric("Classes",             "8")
c4.metric("Model",               "Swin-T")

st.markdown("---")

# ==========================================================
# UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "📁 Upload Skin Lesion Image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload a dermoscopic image for classification.",
)

# ==========================================================
# PROCESS
# ==========================================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analyzing image…"):

        prediction, confidence, probs = predict(image, model, idx2label)
        top_predictions              = get_top_predictions(probs, idx2label)
        probability_dict             = get_probability_dict(probs, idx2label)

        temp_image_path = os.path.join(BASE_DIR, "temp", "uploaded_image.png")
        image.save(temp_image_path)

        input_tensor = prepare_input(image)
        heatmap      = generate_gradcam(model, image, input_tensor)

        heatmap_path = os.path.join(BASE_DIR, "generated_heatmaps", "latest_heatmap.png")
        Image.fromarray(heatmap).save(heatmap_path)

    full_name            = DISEASE_FULL_NAMES.get(prediction, prediction)
    risk_label, risk_type = RISK_LEVEL.get(prediction, ("⚪ Unknown", "info"))
    risk_css = {
        "error":   "risk-high",
        "warning": "risk-medium",
        "success": "risk-low",
    }.get(risk_type, "risk-low")

    # ======================================================
    # RESULT SUMMARY
    # ======================================================

    st.markdown("## 📊 Analysis Results")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("**Uploaded Image**")
        st.image(image, use_container_width=True)

    with col2:

        st.markdown("**Diagnosis**")

        if risk_type == "error":
            st.error(f"🩺 **{full_name}** ({prediction})")
        elif risk_type == "warning":
            st.warning(f"🩺 **{full_name}** ({prediction})")
        else:
            st.success(f"🩺 **{full_name}** ({prediction})")

        st.markdown(
            f'<span class="risk-badge {risk_css}">{risk_label}</span>',
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown("**Prediction Confidence**")
        st.progress(confidence, text=f"{confidence*100:.2f}%")

        st.markdown("")
        st.markdown("**About this condition**")
        st.info(DISEASE_INFO.get(prediction, "No information available."))

    st.markdown("---")

    # ======================================================
    # TOP PREDICTIONS TABLE
    # ======================================================

    st.markdown("## 🔍 Top 3 Predictions")

    top_df = pd.DataFrame(top_predictions)
    top_df.columns = ["Disease Code", "Confidence (%)"]
    top_df["Full Name"]       = top_df["Disease Code"].map(DISEASE_FULL_NAMES)
    top_df["Confidence (%)"]  = top_df["Confidence (%)"].map(lambda x: f"{x:.2f}%")
    top_df = top_df[["Disease Code", "Full Name", "Confidence (%)"]]

    st.dataframe(top_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ======================================================
    # PROBABILITY CHART
    # ======================================================

    st.markdown("## 📈 Full Probability Distribution")

    chart_df = pd.DataFrame({
        "Disease":     list(probability_dict.keys()),
        "Probability": list(probability_dict.values()),
    }).set_index("Disease")

    st.bar_chart(chart_df)

    st.markdown("---")

    # ======================================================
    # GRAD-CAM SIDE BY SIDE
    # ======================================================

    st.markdown("## 🔬 AI Explainability — Grad-CAM")
    st.caption(
        "Grad-CAM highlights the image regions most influential "
        "to the model's prediction. Warmer colours = higher attention."
    )

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Original**")
        st.image(image, use_container_width=True)
    with g2:
        st.markdown("**Grad-CAM Heatmap**")
        st.image(heatmap, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # IMAGE DETAILS (collapsible)
    # ======================================================

    with st.expander("🖼️ Image Details"):
        w, h = image.size
        st.markdown(f"- **Filename:** {uploaded_file.name}")
        st.markdown(f"- **Dimensions:** {w} × {h} px")
        st.markdown(f"- **File size:** {uploaded_file.size / 1024:.1f} KB")
        st.markdown(f"- **Mode:** {image.mode}")

    # ======================================================
    # PDF REPORT
    # ======================================================

    st.markdown("## 📄 Clinical Report")
    st.caption("Download a formatted PDF summary of this analysis.")

    pdf_path = os.path.join(
        BASE_DIR, "generated_reports", "DermaVision_Report.pdf"
    )

    create_pdf_report(
        pdf_path=pdf_path,
        prediction=prediction,
        full_name=full_name,
        confidence=confidence,
        risk_label=risk_label,
        uploaded_image_path=temp_image_path,
        heatmap_path=heatmap_path,
        probabilities=probability_dict,
        filename=uploaded_file.name,
    )

    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_file,
            file_name="DermaVision_Report.pdf",
            mime="application/pdf",
        )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")
st.markdown(
    '<p class="footer-cap">DermaVision v2.0 &nbsp;|&nbsp; '
    "Swin Transformer Tiny &nbsp;|&nbsp; ISIC 2019 &nbsp;|&nbsp; "
    "Educational use only — not a medical device.</p>",
    unsafe_allow_html=True,
)