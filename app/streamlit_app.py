"""MammoAI — Breast Cancer Image Classification Streamlit App."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.components.eda_charts import render_eda_tab
from app.components.model_metrics import render_metrics_tab
from app.components.prediction import render_prediction_tab
from app.components.sidebar import render_sidebar
from app.utils.inference import model_is_ready, predict

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MammoAI — Breast Cancer Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ──────────────────────────────────────────────────────────
css_path = ROOT / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="mammo-header">
        <div>
            <h1>MammoAI</h1>
            <div class="mammo-subtitle">
                Breast Cancer Detection &nbsp;·&nbsp; Mammography Image Classification &nbsp;·&nbsp;
                EfficientNetB0 Transfer Learning
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Disclaimer ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="disclaimer">
        ⚕ <span><b>Educational purposes only.</b> This tool is a portfolio demonstration and is
        <b>not</b> a certified medical diagnostic device. Always consult a qualified physician
        for any clinical decision.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar → image upload ─────────────────────────────────────────────────────
image, predict_clicked = render_sidebar()

# ── Session state ──────────────────────────────────────────────────────────────
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "last_image_id" not in st.session_state:
    st.session_state.last_image_id = None

# Clear result when a new image is uploaded
current_image_id = id(image) if image is not None else None
if current_image_id != st.session_state.last_image_id:
    st.session_state.prediction_result = None
    st.session_state.last_image_id = current_image_id

if predict_clicked and image is not None:
    if not model_is_ready():
        st.sidebar.error(
            "Model not found. Run `notebooks/breast_cancer_analysis.ipynb` "
            "to train and save the model."
        )
    else:
        with st.spinner("Analysing image…"):
            st.session_state.prediction_result = predict(image)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_pred, tab_eda, tab_metrics, tab_about = st.tabs([
    "🔬 Prediction",
    "📊 EDA & Dataset",
    "📈 Model Performance",
    "ℹ About",
])

with tab_pred:
    render_prediction_tab(st.session_state.prediction_result, image)

with tab_eda:
    render_eda_tab()

with tab_metrics:
    render_metrics_tab()

with tab_about:
    st.markdown('<div class="section-header">About MammoAI</div>', unsafe_allow_html=True)

    col_info, col_tech = st.columns([3, 2])

    with col_info:
        st.markdown("""
        **MammoAI** is a portfolio machine learning project demonstrating end-to-end
        image classification for breast cancer detection using mammography images.
        A pre-trained **EfficientNetB0** backbone is fine-tuned on the dataset to classify
        mammograms as **malignant** or **benign**.

        **Workflow:**
        1. Exploratory data analysis of the mammography image dataset in Jupyter
        2. Two-phase transfer learning: frozen backbone → fine-tune top 30 layers
        3. Model evaluation: AUC-ROC, confusion matrix, PR curve on held-out test set
        4. Interactive Streamlit app for real-time image upload and inference
        """)

        st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Source:** [Kaggle — Breast Cancer Detection (hayder17)](https://www.kaggle.com/datasets/hayder17/breast-cancer-detection)
        - **Type:** Mammography images, preprocessed via Roboflow
        - **Size:** ~3,383 images at 640 × 640 px
        - **Split:** Train / Validation / Test (Roboflow YOLO export)
        - **Classes:** Benign, Malignant
        - **Format:** JPG / PNG with YOLO bounding-box annotations
        """)

    with col_tech:
        st.markdown('<div class="section-header">Tech Stack</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="badge-row">
                <span class="tech-badge">TensorFlow</span>
                <span class="tech-badge">Keras</span>
                <span class="tech-badge">EfficientNetB0</span>
                <span class="tech-badge">Transfer Learning</span>
                <span class="tech-badge">Streamlit</span>
                <span class="tech-badge">Plotly</span>
                <span class="tech-badge">OpenCV</span>
                <span class="tech-badge">Pillow</span>
                <span class="tech-badge">Jupyter</span>
                <span class="tech-badge">Python 3.10+</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-header">Model Summary</div>', unsafe_allow_html=True)
        st.markdown("""
        | Component        | Detail                       |
        |------------------|------------------------------|
        | Backbone         | EfficientNetB0 (ImageNet)    |
        | Input size       | 224 × 224 × 3                |
        | Fine-tune        | Top 30 layers                |
        | Head             | GAP → Dense 256 → Dense 128  |
        | Output           | Sigmoid (binary)             |
        | Augmentation     | Flip, rotate, brightness     |
        """)

        st.markdown('<div class="section-header">Author</div>', unsafe_allow_html=True)
        st.markdown("""
        Built by **NULL_IS_OKZ**
        Portfolio project — [GitHub](https://github.com/NULL_IS_OKZ)
        """)

    st.markdown("---")
    st.markdown(
        """
        <div class="disclaimer">
            ⚕ <b>Disclaimer:</b> This application is for educational and portfolio demonstration
            purposes only. It is <b>not</b> a medical device and should not be used for clinical
            diagnosis or treatment decisions. Always consult a licensed healthcare professional.
        </div>
        """,
        unsafe_allow_html=True,
    )
