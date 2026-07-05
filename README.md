# MammoAI

A portfolio machine learning project for breast cancer detection from mammography images. An **EfficientNetB0** backbone is fine-tuned via transfer learning to classify mammograms as **benign** or **malignant**, served through an interactive **Streamlit** app.

> ⚕ **Educational purposes only.** This is a portfolio demonstration, not a certified medical diagnostic device. Do not use it for clinical decisions — always consult a qualified physician.

## What's in the app

Run `streamlit run app/streamlit_app.py` to get a four-tab interface:

- **🔬 Prediction** — upload a mammogram (PNG/JPG/BMP/TIFF) and get a benign/malignant classification with a confidence score, risk level, and a per-class probability chart.
- **📊 EDA & Dataset** — class distribution (overall and per split), and sample images for each class pulled straight from `data/`.
- **📈 Model Performance** — accuracy/AUC/precision/recall/F1 on the held-out test set, confusion matrix, ROC curve, precision-recall curve, and the training history (loss/accuracy per epoch).
- **ℹ About** — dataset source, model architecture, and tech stack.

## Dataset

- **Source:** [Kaggle — Breast Cancer Detection (hayder17)](https://www.kaggle.com/datasets/hayder17/breast-cancer-detection)
- **Size:** ~3,383 mammography images at 640×640 px, preprocessed via Roboflow
- **Classes:** Benign (`0/`), Malignant (`1/`)
- **Split:** Train (2,372) / Validation (675) / Test (336)

Not included in this repo (gitignored) — download it yourself:

```bash
pip install kaggle
kaggle datasets download -d hayder17/breast-cancer-detection -p data --unzip
```

Expected layout:

```
data/
  train/  0/ (benign)   1/ (malignant)
  valid/  0/ (benign)   1/ (malignant)
  test/   0/ (benign)   1/ (malignant)
```

## Model

Two-phase transfer learning on top of an ImageNet-pretrained EfficientNetB0:

| Component        | Detail                                             |
|-------------------|-----------------------------------------------------|
| Backbone          | EfficientNetB0 (ImageNet weights)                   |
| Input size        | 224 × 224 × 3                                       |
| Head              | GlobalAveragePooling → Dense(256) → Dense(128) → Dense(1, sigmoid) |
| Phase 1           | Head only, backbone frozen, lr=1e-3                 |
| Phase 2           | Fine-tune top 30 EfficientNet layers, lr=1e-5        |
| Augmentation      | Horizontal/vertical flip, rotation, brightness, contrast |
| Early stopping    | Monitors `val_auc`, restores best weights           |

Trained via [notebooks/breast_cancer_analysis.ipynb](notebooks/breast_cancer_analysis.ipynb) (or the headless [notebooks/train_model.py](notebooks/train_model.py)).

### Current results (test set, 336 images)

| Metric | Value |
|--------|-------|
| Accuracy | 58.3% |
| AUC-ROC | 0.659 |
| Precision | 46.3% |
| Recall | 58.6% |
| F1 | 0.517 |

Honest caveat: fine-tuning (phase 2) overfit almost immediately on this dataset — validation AUC peaked 2 epochs in before early stopping restored those weights — so the final model performs only modestly better than chance. Retraining with a lower fine-tuning learning rate, stronger regularization, or fewer unfrozen layers would likely improve this; it hasn't been re-tuned since this is primarily an end-to-end pipeline demo.

## Setup

```bash
pip install -r requirements.txt
```

Train the model (writes to `models/`):

```bash
cd notebooks
jupyter notebook breast_cancer_analysis.ipynb   # interactive
# or
python train_model.py                            # headless
```

Run the app:

```bash
streamlit run app/streamlit_app.py
```

## Tech stack

TensorFlow · Keras · EfficientNetB0 · Streamlit · Plotly · OpenCV · Pillow · Jupyter · Python 3.10+

## Project structure

```
mammoai/
  app/                  Streamlit app (tabs, inference, sidebar)
  notebooks/            Training notebook + headless script
  models/               Trained model, metrics, charts (generated)
  data/                 Dataset (gitignored, download separately)
  assets/               Custom CSS
```

## Author

Built by **NULL_IS_OKZ** — [GitHub](https://github.com/nullisokz)
