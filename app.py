# app.py — Face-ID demo (Streamlit)
# Sidebar shows ONLY test accuracy (labelled "Accuracy").
# Models: MobileNetV2 (fine-tuned), SGD-Logistic, SGD-SVM, MLP/ANN, KNN (cosine).
# Classical models consume 1280-D embeddings from MobileNetV2 (conv body + GAP).

import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
import joblib
import pandas as pd
from collections import OrderedDict
from torchvision.models.mobilenetv2 import MobileNetV2
import torch

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

# ── Preprocessing (must match training) ────────────────────────────────────
IMG_SIZE = (160, 160)
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ── Robust loaders ────────────────────────────────────────────────────────
# replace your current helper with this

from torchvision.models.mobilenetv2 import MobileNetV2
import torch

def torch_load_any(p: Path):
    """Load torch object under PyTorch 2.6+ safe loading rules."""
    # allowlist MobileNetV2 so weights_only=True can unpickle it
    try:
        torch.serialization.add_safe_globals([MobileNetV2])
    except Exception:
        pass

    # try safe load first
    try:
        return torch.load(p, map_location=device, weights_only=True)
    except Exception:
        # fallback: only do this if the file is from a trusted source (your own training)
        return torch.load(p, map_location=device, weights_only=False)


def load_finetuned_mobilenet(n_classes: int):
    """
    Load MobileNetV2 fine-tuned model from a .pkl file.
    Handles two cases:
      1) file contains a state_dict (OrderedDict or dict with 'state_dict')
      2) file contains a full nn.Module
    """
    cand = MODELS_DIR / "mobilenet_v2_best.pkl"
    if not cand.exists():
        st.error(f"Missing model file: {cand}")
        st.stop()

    obj = torch_load_any(cand)

    # Build a base MobileNetV2 with correct classifier shape
    def build_blank(nc: int):
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.last_channel, nc)
        return m

    if isinstance(obj, nn.Module):
        model = obj
        model.to(device).eval()
        # (Optional) sanity: if out_features mismatches classes, replace head
        try:
            out = model.classifier[1].out_features
            if out != n_classes:
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
        except Exception:
            pass
        return model

    # If it's a dict / state_dict, load into a fresh backbone with right head
    model = build_blank(n_classes)
    if isinstance(obj, (dict, OrderedDict)):
        state = obj.get("state_dict", obj)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Not fatal; just proceed
    else:
        st.error("Unsupported checkpoint format for MobileNetV2.")
        st.stop()

    model.to(device).eval()
    return model

def load_sklearn_pickle(name: str):
    p = MODELS_DIR / name
    if not p.exists():
        st.error(f"Missing model file: {p}")
        st.stop()
    return joblib.load(p)

# ── Load models (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def load_all_models():
    # CNN head for end-to-end inference
    cnn = load_finetuned_mobilenet(len(CLASS_NAMES))

    # Feature extractor: conv body only; produce 1280-D via GAP
    extractor = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    extractor.classifier = nn.Identity()
    extractor.to(device).eval()

    # Classical estimators saved as sklearn pipelines
    log_model = load_sklearn_pickle("logistic_sgd_best.pkl")
    svm_model = load_sklearn_pickle("svm_best.pkl")
    knn_model = load_sklearn_pickle("knn_best.pkl")
    mlp_model = load_sklearn_pickle("mlp_embeddings_tuned_best.pkl")

    return extractor, cnn, log_model, svm_model, knn_model, mlp_model

extractor, cnn_model, log_model, svm_model, knn_model, mlp_model = load_all_models()

@torch.no_grad()
def embed_1280(x: torch.Tensor) -> np.ndarray:
    """Run conv body → AdaptiveAvgPool2d → flatten to (B,1280)."""
    f = extractor(x)                 # (B, 1280, H', W')
    f = F.adaptive_avg_pool2d(f, (1, 1))
    f = torch.flatten(f, 1)          # (B, 1280)
    return f.cpu().numpy()

# ── UI ─────────────────────────────────────────────────────────────────────
st.title("🔐 Face-ID Classifier")

uploaded = st.file_uploader("Upload a face image", type=["png", "jpg", "jpeg"])
model_choice = st.selectbox(
    "Choose your model",
    ["MobileNetV2 (ft)", "SGD-Logistic", "SGD-SVM", "MLP / ANN", "KNN (cosine)"],
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your input", use_column_width=True)

    x = preprocess(img).unsqueeze(0).to(device)

    if model_choice == "MobileNetV2 (ft)":
        with torch.no_grad():
            logits = cnn_model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            idx = int(np.argmax(probs))
        st.success(f"🏷️ Predicted identity: **{CLASS_NAMES[idx]}**")
    else:
        # Classical models: first get 1280-D embeddings
        feats = embed_1280(x)
        if model_choice == "SGD-Logistic":
            idx = int(log_model.predict(feats)[0])
        elif model_choice == "SGD-SVM":
            idx = int(svm_model.predict(feats)[0])
        elif model_choice == "MLP / ANN":
            idx = int(mlp_model.predict(feats)[0])
        else:
            idx = int(knn_model.predict(feats)[0])
        st.success(f"🏷️ Predicted identity: **{CLASS_NAMES[idx]}**")

st.caption("Sidebar shows **test** accuracy only. Models load from ./models. Class list from ./data/processed/train.")
