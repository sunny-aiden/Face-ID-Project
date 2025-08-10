# app.py — Face-ID demo (Streamlit)
# Sidebar shows ONLY test accuracy (labelled "Accuracy").
# Models: MobileNetV2 (fine-tuned), SGD-Logistic, SGD-SVM, MLP/ANN, KNN (cosine).
# Classical models consume 1280-D embeddings from MobileNetV2 (conv body + GAP).

from __future__ import annotations

import io
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFile
from torch import nn
from torchvision import models, transforms
from torchvision.models.mobilenetv2 import MobileNetV2
from torchvision.transforms import InterpolationMode

# Make Pillow tolerant to truncated files (some camera/phone uploads)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Paths & device ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR   = PROJECT_ROOT / "models"              # put your .pkl files here
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
TRAIN_DIR    = DATA_DIR / "train"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Class names (same order as training) ───────────────────────────────────
if not TRAIN_DIR.exists():
    st.error(f"Missing dataset folder: {TRAIN_DIR}")
    st.stop()
CLASS_NAMES = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])

# ── Sidebar: show ONLY test accuracy (rename “Accuracy”) ───────────────────
# Fill with your final post-tuning test accuracies
TEST_ACCURACY = {
    "MobileNetV2 (ft)": 0.848,
    "SGD-Logistic":     0.392,
    "SGD-SVM":          0.384,
    "MLP / ANN":        0.374,
    "KNN (cosine)":     0.268,
}
st.sidebar.header("🔍 Accuracy (Test)")
acc_rows = [{"Model": k, "Accuracy": f"{v:.3f}"} for k, v in TEST_ACCURACY.items()]
acc_df = pd.DataFrame(acc_rows).sort_values("Accuracy", ascending=False)
st.sidebar.dataframe(acc_df, hide_index=True, use_container_width=True)

# ── Image preprocessing (must match training) ──────────────────────────────
# If training used different stats/size (e.g., ImageNet 224 / mean/std),
# update IMG_SIZE and Normalize below accordingly.
IMG_SIZE = (160, 160)
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),                          # -> float32 [0,1]
    transforms.Normalize([0.5, 0.5, 0.5],           # center to [-1,1]
                         [0.5, 0.5, 0.5]),
])

def to_pil_rgb(img_in: Any) -> Image.Image:
    """
    Normalize any input (UploadedFile / bytes / ndarray / tensor / PIL)
    to a fully-loaded, EXIF-corrected PIL RGB image.

    We force-load with .copy() so there is no open file handle or lazy decoder
    left when Streamlit tries to serialize the image.
    """
    if isinstance(img_in, Image.Image):
        img = img_in.copy()  # force load
    elif hasattr(img_in, "read"):  # Streamlit UploadedFile
        img = Image.open(img_in).copy()
    elif isinstance(img_in, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img_in)).copy()
    elif isinstance(img_in, np.ndarray):
        arr = img_in
        if arr.dtype != np.uint8:
            if arr.dtype.kind == "f":
                arr = np.clip(arr, 0, 1) * 255.0
            arr = arr.astype(np.uint8)
        img = Image.fromarray(arr).copy()
    elif isinstance(img_in, torch.Tensor):
        img = transforms.functional.to_pil_image(img_in.detach().cpu()).copy()
    else:
        raise TypeError(f"Unsupported input type: {type(img_in)}")

    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

# ── Robust Torch loaders ───────────────────────────────────────────────────
def torch_load_any(p: Path):
    """
    Load a torch object under safer rules.
    Tries safe loading (weights_only=True) first; falls back if needed.

    Note: torch.serialization.add_safe_globals exists on newer PyTorch.
    On older versions this call is ignored by the try/except.
    """
    try:
        # Allow MobileNetV2 class during safe unpickling (if needed).
        torch.serialization.add_safe_globals([MobileNetV2])
    except Exception:
        pass

    try:
        return torch.load(p, map_location=device, weights_only=True)
    except Exception:
        # Fallback only for trusted checkpoints produced by your own code.
        return torch.load(p, map_location=device, weights_only=False)

def load_finetuned_mobilenet(n_classes: int) -> nn.Module:
    """
    Load MobileNetV2 fine-tuned model from a .pkl file.
    Supports two checkpoint styles:
      (1) full nn.Module; (2) state_dict or {"state_dict": ...}.
    Ensures the classifier head matches the number of classes.
    """
    ckpt = MODELS_DIR / "mobilenet_v2_best.pkl"
    if not ckpt.exists():
        st.error(f"Missing model file: {ckpt}")
        st.stop()

    obj = torch_load_any(ckpt)

    # base with ImageNet weights (swap head later as needed)
    def build_blank(nc: int):
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.last_channel, nc)
        return m

    if isinstance(obj, nn.Module):
        model = obj
        # (optional) repair head if classes changed
        try:
            out = model.classifier[1].out_features
            if out != n_classes:
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
        except Exception:
            pass
        return model.to(device).eval()

    # dict / state_dict path
    model = build_blank(n_classes)
    if isinstance(obj, (dict, OrderedDict)):
        state = obj.get("state_dict", obj)
        model.load_state_dict(state, strict=False)  # tolerate minor key mismatches
        return model.to(device).eval()

    st.error("Unsupported checkpoint format for MobileNetV2.")
    st.stop()

def load_sklearn_pickle(name: str) -> Optional[Any]:
    """
    Load sklearn/joblib artifacts with friendly error messages.
    Returns None if loading fails so the UI can degrade gracefully.
    """
    p = MODELS_DIR / name
    if not p.exists():
        st.warning(f"Missing model file: {p}")
        return None
    try:
        return joblib.load(p)
    except Exception as e:
        st.warning(f"Could not load '{name}': {e}")
        return None

# ── Load models (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def load_all_models():
    # End-to-end CNN
    cnn = load_finetuned_mobilenet(len(CLASS_NAMES))

    # Feature extractor: MobileNetV2 forward with classifier set to Identity
    # still returns the 1280-D pooled vector in torchvision (keeps avgpool).
    extractor = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    extractor.classifier = nn.Identity()
    extractor.to(device).eval()

    # Classical estimators saved as sklearn pipelines
    log_model = load_sklearn_pickle("sgdlog_best.pkl")
    svm_model = load_sklearn_pickle("sgdsvm_best.pkl")
    knn_model = load_sklearn_pickle("knn_best.pkl")
    mlp_model = load_sklearn_pickle("mlp_embeddings_tuned_best.pkl")

    return extractor, cnn, log_model, svm_model, knn_model, mlp_model

extractor, cnn_model, log_model, svm_model, knn_model, mlp_model = load_all_models()

@torch.no_grad()
def embed_1280(x: torch.Tensor) -> np.ndarray:
    """
    Get 1280-D embeddings from MobileNetV2.
    If extractor returns (B, 1280), use it.
    If it returns (B, 1280, H, W), pool+flatten.
    """
    z = extractor(x)
    if not isinstance(z, torch.Tensor):
        raise RuntimeError("Extractor did not return a tensor.")
    if z.ndim == 2 and z.shape[1] == 1280:
        return z.cpu().numpy()
    if z.ndim == 4 and z.shape[1] == 1280:
        z = F.adaptive_avg_pool2d(z, (1, 1))
        z = torch.flatten(z, 1)
        return z.cpu().numpy()
    raise RuntimeError(f"Unexpected extractor output shape: {tuple(z.shape)}")

# ── UI ─────────────────────────────────────────────────────────────────────
st.title("🔐 Face-ID Classifier")

uploaded = st.file_uploader("Upload a face image", type=["png", "jpg", "jpeg"])
model_choice = st.selectbox(
    "Choose your model",
    ["MobileNetV2 (ft)", "SGD-Logistic", "SGD-SVM", "MLP / ANN", "KNN (cosine)"],
)

if uploaded:
    # 1) Detach from the upload stream to avoid lazy file handles
    raw_bytes = uploaded.getvalue()

    # 2) Normalize to fully-loaded PIL RGB
    img = to_pil_rgb(raw_bytes)

    # 3) Display robustly (fallback to PNG re-encode if needed)
    try:
        st.image(img, caption="Your input", use_container_width=True)
    except Exception as e:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        st.image(buf.getvalue(), caption="Your input", use_container_width=True)
        st.info(f"(Displayed via PNG re-encode fallback: {e})")

    # 4) Tensorize & normalize (must match training)
    x = preprocess(img).unsqueeze(0).to(device)  # [1, 3, H, W], float32

    if model_choice == "MobileNetV2 (ft)":
        with torch.inference_mode():
            logits = cnn_model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            idx = int(np.argmax(probs))
        label = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else "N/A"
        st.success(f"🏷️ Predicted identity: **{label}**")

    else:
        # Classical models: first get 1280-D embeddings
        feats = embed_1280(x)
        chosen = None
        if model_choice == "SGD-Logistic":
            chosen = log_model
        elif model_choice == "SGD-SVM":
            chosen = svm_model
        elif model_choice == "MLP / ANN":
            chosen = mlp_model
        else:
            chosen = knn_model

        if chosen is None:
            st.warning("Selected classical model is unavailable. Check model files / versions.")
        else:
            try:
                idx = int(chosen.predict(feats)[0])
                label = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else "N/A"
                st.success(f"🏷️ Predicted identity: **{label}**")
            except Exception as e:
                st.error(f"Inference failed for {model_choice}: {e}")

st.caption("Sidebar shows **test** accuracy only. Models load from ./models. Class list from ./data/processed/train.")
