"""Model performance visualization components."""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.utils.inference import load_metrics, load_training_history

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def render_metrics_tab():
    metrics = load_metrics()
    history = load_training_history()
    class_names = metrics.get("class_names", ["benign", "malignant"])

    if not metrics:
        st.warning(
            "Model metrics not found. Run `notebooks/breast_cancer_analysis.ipynb` "
            "to train the model and generate metrics."
        )
        _render_architecture_card()
        return

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Test Set Performance</div>', unsafe_allow_html=True)

    cards = [
        ("Accuracy",  metrics.get("accuracy", 0),  "blue",  ".2%"),
        ("AUC-ROC",   metrics.get("auc", 0),        "green", ".4f"),
        ("Precision", metrics.get("precision", 0),  "amber", ".2%"),
        ("Recall",    metrics.get("recall", 0),      "red",   ".2%"),
        ("F1 Score",  metrics.get("f1", 0),          "blue",  ".4f"),
    ]
    cols = st.columns(len(cards))
    for col, (label, value, color, fmt) in zip(cols, cards):
        col.markdown(
            f"""<div class="metric-card">
                <div class="label">{label}</div>
                <div class="value {color}">{value:{fmt}}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion matrix + ROC ────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = np.array(metrics.get("confusion_matrix", [[0, 0], [0, 0]]))
        n  = cm.shape[0]
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.where(row_sums > 0, cm.astype(float) / row_sums, 0)
        text_labels = [[f"{v}<br>({p:.1%})" for v, p in zip(row_v, row_p)]
                       for row_v, row_p in zip(cm, cm_pct)]

        axis_labels = [c.title() for c in class_names]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=[f"Pred: {l}" for l in axis_labels],
            y=[f"True: {l}" for l in axis_labels],
            text=text_labels,
            texttemplate="%{text}",
            textfont={"size": 15},
            colorscale="Blues",
            showscale=False,
        ))
        fig_cm.update_layout(
            paper_bgcolor="#0f1117",
            font={"color": "#e8eaf0", "size": 12},
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        fpr = metrics.get("roc_fpr", [0, 1])
        tpr = metrics.get("roc_tpr", [0, 1])
        auc = metrics.get("auc", 0)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"AUC = {auc:.4f}",
            line=dict(color="#1E88E5", width=2.5),
            fill="tozeroy", fillcolor="rgba(30,136,229,0.1)",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random",
            line=dict(color="#4a5568", width=1.5, dash="dash"),
        ))
        fig_roc.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1a2035",
            font={"color": "#e8eaf0"},
            height=300,
            margin=dict(l=10, r=10, t=10, b=40),
            xaxis={"title": "False Positive Rate", "gridcolor": "#2a3350"},
            yaxis={"title": "True Positive Rate",  "gridcolor": "#2a3350"},
            legend={"font": {"color": "#e8eaf0"}},
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Training history + PR curve ───────────────────────────────────────────
    col_left3, col_right3 = st.columns(2)

    with col_left3:
        st.markdown('<div class="section-header">Training History</div>', unsafe_allow_html=True)
        if history:
            epochs = list(range(1, len(history.get("loss", [])) + 1))
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=epochs, y=history.get("accuracy", []),
                name="Train Accuracy", line=dict(color="#1E88E5", width=2),
            ))
            fig_hist.add_trace(go.Scatter(
                x=epochs, y=history.get("val_accuracy", []),
                name="Val Accuracy", line=dict(color="#42A5F5", width=2, dash="dash"),
            ))
            fig_hist.add_trace(go.Scatter(
                x=epochs, y=history.get("loss", []),
                name="Train Loss", line=dict(color="#E53935", width=2),
                yaxis="y2",
            ))
            fig_hist.add_trace(go.Scatter(
                x=epochs, y=history.get("val_loss", []),
                name="Val Loss", line=dict(color="#EF9950", width=2, dash="dash"),
                yaxis="y2",
            ))
            fig_hist.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#1a2035",
                font={"color": "#e8eaf0"},
                height=300,
                margin=dict(l=10, r=60, t=10, b=40),
                xaxis={"title": "Epoch", "gridcolor": "#2a3350"},
                yaxis={"title": "Accuracy", "gridcolor": "#2a3350", "range": [0, 1.05]},
                yaxis2={
                    "title": "Loss", "overlaying": "y", "side": "right",
                    "showgrid": False,
                },
                legend={"font": {"color": "#e8eaf0"}, "x": 0.01, "y": 0.02},
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            # Fall back to saved PNG
            png = MODELS_DIR / "training_curves.png"
            if png.exists():
                st.image(str(png), use_container_width=True)
            else:
                st.info("Training history not available. Run the notebook to generate it.")

    with col_right3:
        st.markdown('<div class="section-header">Precision-Recall Curve</div>', unsafe_allow_html=True)
        pr_prec = metrics.get("pr_precision", [])
        pr_rec  = metrics.get("pr_recall", [])
        if pr_prec and pr_rec:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=pr_rec, y=pr_prec, mode="lines", name="PR Curve",
                line=dict(color="#E53935", width=2.5),
                fill="tozeroy", fillcolor="rgba(229,57,53,0.1)",
            ))
            fig_pr.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#1a2035",
                font={"color": "#e8eaf0"},
                height=300,
                margin=dict(l=10, r=10, t=10, b=40),
                xaxis={"title": "Recall",    "gridcolor": "#2a3350", "range": [0, 1]},
                yaxis={"title": "Precision", "gridcolor": "#2a3350", "range": [0, 1.05]},
            )
            st.plotly_chart(fig_pr, use_container_width=True)
        else:
            png = MODELS_DIR / "evaluation_charts.png"
            if png.exists():
                st.image(str(png), use_container_width=True)
            else:
                st.info("PR data not available. Run the notebook to generate it.")

    st.markdown("<br>", unsafe_allow_html=True)
    _render_architecture_card()


def _render_architecture_card():
    with st.expander("Model Architecture", expanded=False):
        st.markdown("""
        **Base:** EfficientNetB0 (pretrained on ImageNet, fine-tuned top 30 layers)

        ```
        Input (224 × 224 × 3)
            │
            ├─ EfficientNetB0 backbone  (feature extractor)
            ├─ GlobalAveragePooling2D
            ├─ Dense(256, relu) → BatchNormalization → Dropout(0.5)
            ├─ Dense(128, relu) → Dropout(0.3)
            └─ Dense(1, sigmoid)  →  Malignancy Probability
        ```

        | Component        | Detail                                              |
        |------------------|-----------------------------------------------------|
        | Backbone         | EfficientNetB0 (ImageNet weights)                   |
        | Fine-tune layers | Top 30 layers of EfficientNetB0                     |
        | Phase 1 LR       | 1e-3 (head only, base frozen)                       |
        | Phase 2 LR       | 1e-5 (fine-tuning)                                  |
        | Loss             | Binary Cross-Entropy                                |
        | Early Stopping   | patience=15, monitor val_auc, restore best weights  |
        | Input size       | 224 × 224 px                                        |
        | Batch size       | 32                                                  |
        | Augmentation     | H/V flip, rotation, brightness, contrast            |
        """)
