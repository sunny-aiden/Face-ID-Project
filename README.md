# Face-ID-Project

## 8) Team Member Roles & Contributions

| Role | Name | Key Contributions |
|------|------|-------------------|
| **Data & Cleaning** | Yorbis Daniel Alarcon | LFW download/verify, RGB/resize, identity-wise split |
| **EDA & Reporting** | Ramandeep Kaur | Class distribution, brightness/contrast, visuals |
| **Modeling – ML** | All members | Embeddings cache, SGD-Log/SVM/KNN/MLP training & tuning |
| **Modeling – CNN & App** | Sangsun Lee | Optuna (TPE) study for CNN — search over lr, dropout, weight decay, unfreeze with pruning; manual/random grids for ML (SGD-Logistic & SGD-SVM α/η₀/schedule, KNN k/metric, MLP PCA/hidden/α/lr/batch); validation ranking, best-model checkpoints, consolidated scoreboard. |


| **Hyperparameter Tuning** | All members | Class distribution, brightness/contrast, visuals |

| Role                      | Name                                | Key Contributions                                                                                                                                                                                                                                                                                        |
| ------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data & Cleaning**       | **Yorbis Daniel Alarcon**           | LFW acquisition & integrity checks; RGB conversion and 160×160 resize; identity-wise 70/15/15 split; cleaning log JSON.                                                                                                                                                                                  |
| **EDA & Reporting**       | **Ramandeep Kaur**                  | Class distribution (long-tail) analysis; brightness/contrast histograms; sample grids; figures and narrative for report.                                                                                                                                                                                 |
| **Modeling — ML**         | **All members**                     | MobileNetV2 embedding cache; baselines: SGD-Logistic, SGD-SVM, KNN, and MLP (ANN); training/evaluation scripts and metrics.                                                                                                                                                                              |
| **Modeling — CNN**        | **Sangsun Lee**                     | MobileNetV2 fine-tuning (partial unfreeze + dropout); checkpoints, early stopping, result tracking.                                                                                                                                                                                                      |
| **Hyperparameter Tuning** | **All members ** | Optuna (TPE) study for CNN — search over **lr**, **dropout**, **weight decay**, **unfreeze depth** with pruning; manual/random grids for ML (SGD-Logistic & SGD-SVM: α/η₀/schedule; KNN: *k*/metric; MLP: PCA/hidden/α/lr/batch); validation ranking; best-model checkpointing; consolidated scoreboard. |
| **Demo & Deployment**     | **Krunal Patel**     | Streamlit app (image upload, model selector, prediction + **test accuracy** in sidebar); safe model loading (torch/joblib); packaging (`requirements.txt`, `.python-version`), GitHub setup, and Streamlit Community Cloud deployment guidance. |


## Dataset - Face Identification on LFW — Classical ML vs Fine-tuned CNN

Practical face identification built on the **Labeled Faces in the Wild (LFW)** dataset. This project combines classical machine learning baselines trained on frozen deep embeddings with an end-to-end fine-tuned MobileNetV2 CNN to handle the challenges of many identities with few examples and in-the-wild variability. Dataset can be downloaded here: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

---

## 1) Project Goals & Overview

Given a face image, predict its identity among people seen during training. LFW is difficult due to severe class imbalance and small per-identity sample counts. We tackle this via two complementary tracks:

### A. Classical ML on Embeddings
- Images are resized/aligned to **160×160** and passed through a pretrained **MobileNetV2 (ImageNet)** to extract fixed embeddings (global average pooled).
- On top of those embeddings we train:
  - **SGD Logistic Regression** (`sgd-log`)
  - **SGD Linear SVM** (`sgd-svm`)
  - **K-NN** (cosine / euclidean as explored)
  - **MLP / ANN** (on PCA-reduced embeddings)
- Fast, lightweight baselines saved with `joblib.dump`.

### B. Fine-tuned CNN
- Start from pretrained **MobileNetV2**.
- Unfreeze a subset of layers and add dropout; fine-tune end-to-end with augmentation.
- Hyperparameters (learning rate, weight decay, dropout, unfreeze depth) are searched with **Optuna (TPE)** with early pruning/early stopping.
- Final model is saved as PyTorch weights.

---

## 2) Dataset (LFW) Details

- **Name:** Labeled Faces in the Wild (LFW)  
- **Scale:** ~13,233 images over ~5,749 people; ~**1,680** people have 2+ images after filtering.  
- **Challenges:** Heavy class imbalance; many identities with very few samples; unconstrained “in the wild” conditions (pose, lighting, occlusion).

**Preprocessing pipeline includes:**
- Download and clean/split (train/val/test) with fixed seed.
- Standardize to **RGB** and **160×160** resolution (PNG/JPG).
- Consistent class indexing across splits (empty class folders preserved to avoid label shifts).
- Data augmentation for CNN: random horizontal flip, small rotation, color jitter; normalize to mean=0.5, std=0.5.

---

## 3) Models & Training

### Classical Models (on 1280-D deep embeddings)
- `StandardScaler` before SGD models; **PCA(128/256, whiten)** for MLP.
- Small manual/randomized search over `alpha`, `eta0`, learning-rate schedule (SGD variants), and `k/metric` (KNN).

### CNN Fine-tuning
- `torchvision.models.mobilenet_v2` (ImageNet weights).
- Replace head with `Dropout + Linear(last_channel → num_classes)`.
- Partially unfreeze tail (Optuna choice). Optimizer: **AdamW**.
- Early stopping on validation accuracy; best checkpoint retained.

---

## 4) Results Summary

| Model                         | Validation Acc. | Test Acc. | Notes |
|------------------------------|-----------------|-----------|-------|
| **MobileNetV2 (fine-tuned)** | **0.840**       | **0.848** | Optuna-tuned LR/WD/dropout/unfreeze; early stop. |
| SGD-Logistic (embeddings)    | 0.387           | 0.392     | Multinomial, SAGA on scaled features. |
| SGD-SVM (hinge, embeddings)  | 0.407           | 0.384     | Linear margin baseline. |
| **MLP / ANN** (embeddings)   | 0.417           | 0.374     | StdScaler → PCA → MLP(512), ReLU. |
| K-NN (cosine, embeddings)    | 0.265           | 0.268     | Non-parametric baseline. |

**Rank:** **CNN ≫ (SVM ≈ MLP ≈ Logistic) ≫ KNN**  
**Takeaway:** Domain-adapted features from fine-tuning dominate frozen-feature baselines.

---

## 5) Repository Structure

```
Face-ID-Project/
├── data/
│   └── processed/
│       ├── train/                # ImageFolder: one subfolder per identity
│       ├── val/
│       └── test/
├── features/                     # cached deep embeddings: X_*.npy, y_*.npy
├── models/
│   ├── mobilenet_v2_best.pkl
│   ├── logistic_sgd_best.pkl
│   ├── svm_best.pkl
│   ├── knn_best.pkl
│   ├── mlp_embeddings_tuned_best.pkl
│   └── metrics/
│       └── all_models_scoreboard.txt
├── notebooks/                    # EDA / cleaning / modeling / tuning
├── app.py                        # Streamlit demo
└── README.md
```

---

## 6) Setup & Run (Local)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scriptsctivate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install scikit-learn optuna streamlit pillow matplotlib seaborn tqdm joblib
```

**Launch the demo:**
```bash
streamlit run app.py
# if port is busy → streamlit run app.py --server.port 8502
```

*Tip (stop a stuck app):*  
macOS/Linux: `pkill -f "streamlit run app.py"` or `lsof -ti:8501 | xargs kill -9`  
Windows: find PID via `netstat -ano | findstr :8501` then `taskkill /PID <PID> /F`

*PyTorch ≥ 2.6 note:* model loading uses safe unpickling and falls back to `weights_only=False` only for trusted checkpoints.

---

## 7) How to Reproduce (Short)

1. **Clean & split:** verify images → RGB → resize 160×160 → `data/processed/{train,val,test}` by identity (fixed seed).  
2. **Embeddings:** MobileNetV2 conv body + GAP → save `features/X_*.npy, y_*.npy`.  
3. **Classical models:** train/tune SGD-Logistic, SGD-SVM, KNN, **MLP**; save `*.pkl`.  
4. **Fine-tune CNN:** replace head, unfreeze tail, augment; tune with Optuna; save `mobilenet_v2_best.pkl`.  
5. **Demo:** `streamlit run app.py`, upload face image, choose model.

---


