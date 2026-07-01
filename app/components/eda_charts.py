"""EDA visualization components for image dataset."""
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from app.utils.inference import load_class_distribution, load_app_config

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def _get_sample_images(cls_name: str, n: int = 6) -> list[Image.Image]:
    """Return up to n sample images for a given class.

    Dataset uses numeric subfolders: 0/ = benign, 1/ = malignant.
    """
    folder_map = {"benign": "0", "malignant": "1"}
    folder = folder_map.get(cls_name.lower(), cls_name)
    images = []
    for split in ["train", "valid"]:
        img_dir = DATA_ROOT / split / folder
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except Exception:
                continue
            if len(images) >= n:
                return images
    return images


def render_eda_tab():
    dist = load_class_distribution()
    cfg  = load_app_config()
    class_names = cfg.get("class_names", ["benign", "malignant"])

    has_data = DATA_ROOT.exists() and any(
        (DATA_ROOT / s / "images").exists() for s in ["train", "valid", "test"]
    )

    # ── Row 1: Class distribution ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    if not dist:
        st.warning(
            "Class distribution data not found. Run the Jupyter notebook first, "
            "or ensure `models/class_distribution.json` exists."
        )
    else:
        col1, col2 = st.columns(2)

        with col1:
            all_counts = dist.get("all", {})
            labels = list(all_counts.keys())
            values = list(all_counts.values())
            colors = []
            for lbl in labels:
                if "malignant" in lbl.lower():
                    colors.append("#EF5350")
                elif "benign" in lbl.lower():
                    colors.append("#66BB6A")
                else:
                    colors.append("#42A5F5")

            fig_donut = go.Figure(go.Pie(
                labels=[l.title() for l in labels],
                values=values,
                hole=0.55,
                marker_colors=colors,
                textinfo="label+percent",
                textfont_size=14,
                pull=[0.04] * len(labels),
            ))
            fig_donut.update_layout(
                paper_bgcolor="#0f1117",
                font={"color": "#e8eaf0"},
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True,
                legend={"font": {"color": "#e8eaf0"}},
                annotations=[{
                    "text": f"<b>{sum(values)}</b><br>images",
                    "x": 0.5, "y": 0.5,
                    "font_size": 16, "font_color": "#c5cae9",
                    "showarrow": False,
                }],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col2:
            splits = ["train", "valid", "test"]
            fig_bar = go.Figure()
            split_colors = ["#1E88E5", "#FFCA28", "#43A047"]
            for color, split in zip(split_colors, splits):
                split_data = dist.get(split, {})
                if not split_data:
                    continue
                fig_bar.add_trace(go.Bar(
                    name=split.title(),
                    x=[l.title() for l in split_data.keys()],
                    y=list(split_data.values()),
                    marker_color=color,
                    opacity=0.85,
                    text=list(split_data.values()),
                    textposition="auto",
                ))
            fig_bar.update_layout(
                paper_bgcolor="#0f1117",
                plot_bgcolor="#1a2035",
                font={"color": "#e8eaf0"},
                height=300,
                margin=dict(l=10, r=10, t=10, b=40),
                barmode="group",
                xaxis={"gridcolor": "#2a3350"},
                yaxis={"gridcolor": "#2a3350", "title": "Image Count"},
                legend={"font": {"color": "#e8eaf0"}},
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Row 2: Sample image grid ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Sample Images per Class</div>', unsafe_allow_html=True)

    if not has_data:
        st.warning(
            "Dataset not found in `data/`. "
            "Download from [Kaggle](https://www.kaggle.com/datasets/hayder17/breast-cancer-detection) "
            "and extract into `data/`."
        )
        return

    for cls_name in class_names:
        is_mal = "malignant" in cls_name.lower()
        color  = "#EF5350" if is_mal else "#66BB6A"
        st.markdown(
            f"<div style='font-size:0.9rem;font-weight:700;color:{color};"
            f"text-transform:uppercase;letter-spacing:2px;margin:12px 0 8px 0;'>"
            f"{'⚠ ' if is_mal else '✓ '}{cls_name}</div>",
            unsafe_allow_html=True,
        )
        samples = _get_sample_images(cls_name, n=6)
        if not samples:
            st.caption(f"No samples found for {cls_name}.")
            continue
        cols = st.columns(len(samples))
        for col, img in zip(cols, samples):
            col.image(img, use_container_width=True)

    # ── Row 3: Saved EDA charts from notebook ─────────────────────────────────
    eda_png = MODELS_DIR / "eda_samples.png"
    if eda_png.exists():
        with st.expander("Notebook EDA — Annotated Sample Grid", expanded=False):
            st.image(str(eda_png), use_container_width=True)

    dist_png = MODELS_DIR / "eda_distribution.png"
    if dist_png.exists():
        with st.expander("Notebook EDA — Distribution Charts", expanded=False):
            st.image(str(dist_png), use_container_width=True)
