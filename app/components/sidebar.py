"""Sidebar: image uploader."""
import io

import streamlit as st
from PIL import Image


def render_sidebar() -> tuple[Image.Image | None, bool]:
    """Render the sidebar image uploader. Returns (image, predict_clicked)."""
    st.sidebar.markdown(
        "<div style='text-align:center;padding:8px 0 16px 0;'>"
        "<span style='font-size:1.5rem;font-weight:800;color:#42A5F5;'>Upload Image</span>"
        "<br><span style='color:#8892a4;font-size:0.78rem;'>Mammogram · PNG or JPG</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.sidebar.file_uploader(
        "Choose a mammogram image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Upload a breast mammography image for classification.",
        label_visibility="collapsed",
    )

    image: Image.Image | None = None
    if uploaded is not None:
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        st.sidebar.image(image, caption="Uploaded image", use_container_width=True)

    st.sidebar.markdown("---")

    predict_clicked = st.sidebar.button(
        "Analyse Image",
        use_container_width=True,
        disabled=(image is None),
    )

    if image is None:
        st.sidebar.info("Upload a mammogram to enable analysis.")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='color:#8892a4;font-size:0.75rem;line-height:1.5;'>"
        "<b>Supported formats:</b> PNG, JPG, JPEG, BMP, TIFF<br>"
        "<b>Model input:</b> 224 × 224 px (auto-resized)<br>"
        "<b>Classes:</b> Benign / Malignant"
        "</div>",
        unsafe_allow_html=True,
    )

    return image, predict_clicked
