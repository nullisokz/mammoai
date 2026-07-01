"""Prediction result display for image classification."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from app.utils.inference import load_app_config


def render_prediction_tab(result: dict | None, image: Image.Image | None):
    """Render prediction results: image, badge, confidence, probability bars."""
    if image is None:
        st.markdown(
            "<div style='text-align:center;padding:80px 20px;color:#4a5568;'>"
            "<div style='font-size:4rem;margin-bottom:16px;'>🩻</div>"
            "<div style='font-size:1.15rem;font-weight:600;'>Upload a mammogram image</div>"
            "<div style='font-size:0.85rem;margin-top:8px;color:#6b7280;'>"
            "Use the sidebar to upload a PNG or JPG mammogram, then click <b>Analyse Image</b>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    cfg = load_app_config()
    class_names = cfg["class_names"]

    img_col, result_col = st.columns([1, 1], gap="large")

    with img_col:
        st.markdown('<div class="section-header">Uploaded Mammogram</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        w, h = image.size
        st.caption(f"Original size: {w} × {h} px  ·  Resized to {cfg['img_size']} × {cfg['img_size']} for inference")

    with result_col:
        if result is None:
            st.markdown(
                "<div style='display:flex;align-items:center;justify-content:center;"
                "height:100%;min-height:300px;color:#6b7280;font-size:0.95rem;'>"
                "Click <b>&nbsp;Analyse Image&nbsp;</b> to see results."
                "</div>",
                unsafe_allow_html=True,
            )
            return

        label = result["label"]
        confidence = result["confidence"]
        probs = result["probabilities"]
        is_malignant = result["is_malignant"]
        risk = result["risk"]

        badge_cls   = "malignant" if is_malignant else "benign"
        badge_icon  = "⚠" if is_malignant else "✓"

        st.markdown('<div class="section-header">Classification Result</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="prediction-badge {badge_cls}">
                <div style="font-size:2.8rem;margin-bottom:6px;">{badge_icon}</div>
                <div class="diagnosis-label">{label.upper()}</div>
                <div class="diagnosis-sub">
                    {'Findings suggest malignancy — clinical review recommended'
                     if is_malignant else 'Findings suggest benign tissue — routine follow-up advised'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Metric cards: confidence + risk
        m1, m2 = st.columns(2)
        risk_color = {"High": "red", "Medium": "amber", "Low": "green"}[risk]
        risk_icon  = {"High": "⬆", "Medium": "→", "Low": "⬇"}[risk]

        m1.markdown(
            f"""<div class="metric-card">
                <div class="label">Confidence</div>
                <div class="value {'red' if is_malignant else 'green'}">{confidence:.1%}</div>
                <div style="color:#8892a4;font-size:0.75rem;margin-top:4px;">Model certainty</div>
            </div>""",
            unsafe_allow_html=True,
        )
        m2.markdown(
            f"""<div class="metric-card">
                <div class="label">Risk Level</div>
                <div class="value {risk_color}">{risk_icon} {risk}</div>
                <div style="color:#8892a4;font-size:0.75rem;margin-top:4px;">Based on confidence</div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability bar chart for all classes
        st.markdown('<div class="section-header">Class Probabilities</div>', unsafe_allow_html=True)
        _render_prob_bars(probs, label)


def _render_prob_bars(probs: dict, predicted_label: str):
    class_labels = list(probs.keys())
    values = [probs[c] * 100 for c in class_labels]
    colors = []
    for c in class_labels:
        if "malignant" in c.lower():
            colors.append("rgba(239,83,80,0.85)")
        elif "benign" in c.lower():
            colors.append("rgba(102,187,106,0.85)")
        else:
            colors.append("rgba(66,165,245,0.85)")

    fig = go.Figure(go.Bar(
        x=values,
        y=[c.title() for c in class_labels],
        orientation="h",
        marker_color=colors,
        marker_line_color=[c.replace("0.85", "1") for c in colors],
        marker_line_width=1.5,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont={"size": 13, "color": "#c5cae9"},
    ))
    fig.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a2035",
        font={"color": "#e8eaf0"},
        height=160,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis={
            "range": [0, 110],
            "gridcolor": "#2a3350",
            "ticksuffix": "%",
            "tickfont": {"size": 11},
        },
        yaxis={"gridcolor": "#2a3350", "tickfont": {"size": 13}},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
