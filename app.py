import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import pickle
import joblib

# ── SETTINGS & PATHS ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
CLASS_NAMES  = sorted([d.name for d in (DATA_DIR/"train").iterdir() if d.is_dir()])

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Validation accuracies from your tuning (Step 10 requirement)
METRICS = {
    "Logistic Regression": 0.413,
    "SVM"               : 0.171,
    "KNN"               : 0.247,
    "MobileNetV2"       : 0.840,
}

# ── SIDEBAR METRICS DISPLAY ───────────────────────────────────────────────
st.sidebar.header("🔍 Validation Accuracy")
for name, acc in METRICS.items():
    st.sidebar.write(f"**{name:15s}**  {acc*100:5.1f}%")

# ── TRANSFORMS ───────────────────────────────────────────────────────
IMG_SIZE = (160, 160)
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ── MODEL LOADING ────────────────────────────────────────────────────
@st.cache_resource
def load_models_and_extractor():
    # 1) Fine-tuned MobileNetV2 for end-to-end inference
    def mobilenet_v2_ft(num_classes):
        net = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        # swap out the final linear layer
        net.classifier[1] = nn.Linear(net.last_channel, num_classes)
        return net

    # instantiate model
    mob = mobilenet_v2_ft(len(CLASS_NAMES))

    # candidate checkpoint names (with priority)
    candidate_paths = [
        MODELS_DIR / "mobilenet_v2_best.pth",             # legacy (if renamed)
        MODELS_DIR / "mobilenetv2_best.pth",              # actual file present
        MODELS_DIR / "mobilenetv2_best_checkpoint.pth",    # alternate checkpoint
        MODELS_DIR / "mobilenetv2_full_model.pkl",        # full pickle fallback
    ]

    loaded_mobilenet = False
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            if path.suffix in [".pth", ".pt"]:
                state = torch.load(path, map_location=device)
                # handle possible wrapper
                if isinstance(state, dict) and "state_dict" in state:
                    mob.load_state_dict(state["state_dict"])
                else:
                    # If it's the full model saved instead of state_dict, attempt load
                    try:
                        mob.load_state_dict(state)
                    except Exception:
                        # fallback: maybe it's entire model object
                        mob = state
                st.info(f"Loaded MobileNetV2 from {path.name}")
            elif path.suffix == ".pkl":
                with open(path, "rb") as f:
                    mob = pickle.load(f)
                st.info(f"Loaded full MobileNetV2 object from {path.name}")
            loaded_mobilenet = True
            break
        except Exception as e:
            st.warning(f"Failed to load MobileNet from {path.name}: {e}")
    if not loaded_mobilenet:
        raise FileNotFoundError(
            f"No usable MobileNet checkpoint found in {MODELS_DIR}. "
            f"Contents: {[p.name for p in MODELS_DIR.iterdir()]}"
        )

    mob.to(device).eval()

    # 2) Feature extractor for classical models: mobilenet up to pooling
    feat_ext = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )
    feat_ext.classifier = nn.Identity()
    feat_ext.to(device).eval()

    # 3) Classical models (names adjusted)
    # logistic-like (sgd logistic)
    log_path = MODELS_DIR / "sgdlog_best.pkl"
    if not log_path.exists():
        raise FileNotFoundError(f"Logistic model not found at {log_path}")
    log_model = joblib.load(log_path)

    # svm (sgd svm)
    svm_path = MODELS_DIR / "sgdsvm_best.pkl"
    if not svm_path.exists():
        raise FileNotFoundError(f"SVM model not found at {svm_path}")
    svm_model = joblib.load(svm_path)

    # KNN
    knn_path = MODELS_DIR / "knn_best.pkl"
    if not knn_path.exists():
        raise FileNotFoundError(f"KNN model not found at {knn_path}")
    knn_model = joblib.load(knn_path)

    return feat_ext, mob, log_model, svm_model, knn_model

extractor, mobilenet_model, log_model, svm_model, knn_model = load_models_and_extractor()

# ── STREAMLIT UI ────────────────────────────────────────────────────
st.title("🔐 Face-ID Classifier")

uploaded = st.file_uploader("Upload a face image", type=["png","jpg","jpeg"])
model_choice = st.selectbox("Choose your model",
    ["Logistic Regression","SVM","KNN","MobileNetV2"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Your input", use_column_width=True)

    x = preprocess(img).unsqueeze(0).to(device)

    if model_choice == "MobileNetV2":
        with torch.no_grad():
            logits = mobilenet_model(x)
            if isinstance(logits, torch.Tensor):
                idx = logits.argmax(1).item()
            else:
                # if model is custom object with predict-like interface
                try:
                    idx = int(logits.argmax(1).item())
                except:
                    st.error("Unexpected MobileNet output format.")
                    idx = 0
    else:
        with torch.no_grad():
            feats = extractor(x).cpu().numpy()

        if model_choice == "Logistic Regression":
            idx = log_model.predict(feats)[0]
        elif model_choice == "SVM":
            idx = svm_model.predict(feats)[0]
        else:  # KNN
            idx = knn_model.predict(feats)[0]

    if idx < 0 or idx >= len(CLASS_NAMES):
        st.error(f"Predicted index {idx} out of range.")
    else:
        label = CLASS_NAMES[idx]
        st.success(f"🏷️  Predicted identity: **{label}**")
