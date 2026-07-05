"""Model loading and image inference utilities."""
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

_DEFAULT_CONFIG = {
    "img_size": 224,
    "num_classes": 2,
    "class_names": ["benign", "malignant"],
}


@st.cache_data(show_spinner=False)
def load_app_config() -> dict:
    path = MODELS_DIR / "app_config.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return _DEFAULT_CONFIG


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input

    model_path = MODELS_DIR / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run the Jupyter notebook to train and save the model."
        )

    # Keras 3 cannot reliably deserialize the Lambda preprocessing layer's function
    # from the saved config (it loses the module path). Rebuilding the architecture
    # in code and loading only the weights sidesteps that entirely.
    img_size = load_app_config()["img_size"]
    base = EfficientNetB0(weights=None, include_top=False,
                          input_shape=(img_size, img_size, 3))
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image_input")
    x = layers.Lambda(preprocess_input, output_shape=lambda s: s, name="preprocess")(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.BatchNormalization(name="bn")(x)
    x = layers.Dropout(0.5, name="drop_256")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.3, name="drop_128")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = keras.Model(inputs, outputs, name="MammoAI")
    model.load_weights(str(model_path))
    return model


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    path = MODELS_DIR / "model_metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_training_history() -> dict:
    path = MODELS_DIR / "training_history.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_class_distribution() -> dict:
    path = MODELS_DIR / "class_distribution.json"
    return json.loads(path.read_text()) if path.exists() else {}


def preprocess_image(image: Image.Image, img_size: int = 224) -> np.ndarray:
    """Resize and convert to (1, H, W, 3) float32 array in [0, 255] range.

    The model includes EfficientNet's preprocess_input as a Lambda layer, so
    we must NOT divide by 255 here — the model handles its own normalisation.
    """
    image = image.convert("RGB").resize((img_size, img_size))
    arr = np.array(image, dtype=np.float32)  # [0, 255], no /255
    return np.expand_dims(arr, axis=0)


def predict(image: Image.Image) -> dict:
    """Run inference on a PIL image. Returns label, confidence, probabilities."""
    cfg = load_app_config()
    model = load_model()
    class_names = cfg["class_names"]
    img_size = cfg["img_size"]
    num_classes = cfg["num_classes"]

    arr = preprocess_image(image, img_size)
    raw = model.predict(arr, verbose=0)

    if num_classes == 2:
        mal_prob = float(raw[0][0])
        ben_prob = 1.0 - mal_prob
        probs = {class_names[0]: ben_prob, class_names[1]: mal_prob}
        pred_idx = int(mal_prob >= 0.5)
    else:
        probs = {name: float(p) for name, p in zip(class_names, raw[0])}
        pred_idx = int(np.argmax(raw[0]))

    label = class_names[pred_idx]
    confidence = probs[label]
    is_malignant = "malignant" in label.lower()
    risk = "High" if confidence >= 0.75 else ("Medium" if confidence >= 0.55 else "Low")

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probs,
        "is_malignant": is_malignant,
        "risk": risk,
    }


def model_is_ready() -> bool:
    return (MODELS_DIR / "model.keras").exists()
