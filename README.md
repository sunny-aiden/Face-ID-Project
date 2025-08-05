# Face-ID-Project

## Face Identification on LFW — Classical ML vs Fine-tuned CNN

Practical face identification built on the **Labeled Faces in the Wild (LFW)** dataset. This project combines classical machine learning baselines trained on frozen deep embeddings with an end-to-end fine-tuned MobileNetV2 CNN to handle the challenges of many identities with few examples and in-the-wild variability.

---

## 1) Project Goals & Overview

Given a face image, predict its identity among people seen during training. LFW is difficult due to severe class imbalance and small per-identity sample counts. We tackle this via two complementary tracks:

### A. Classical ML on Embeddings
- Images are resized/aligned to 160×160 and passed through a pretrained MobileNetV2 (ImageNet) to extract fixed embeddings (global average pooled).  
- On top of those embeddings we train:
  - **SGD Logistic Regression** (`sgd-log`)
  - **SGD Linear SVM** (`sgd-svm`)
  - **K-NN** (cosine / euclidean as explored)
- These are fast, lightweight baselines. They use `joblib.dump` for persistence.

### B. Fine-tuned CNN
- Start from pretrained MobileNetV2.
- Unfreeze a subset of layers and add dropout; fine-tune end-to-end with augmentation.
- Hyperparameters (learning rate, weight decay, dropout, unfreeze depth) are searched with **Optuna (TPE)** with early pruning/early stopping.
- Final model is saved as PyTorch weights.

---

## 2) Dataset (LFW) Details

- **Name:** Labeled Faces in the Wild (LFW)  
- **Scale:** ~13,233 images over ~5,749 people; ~1,680 people have 2+ images after filtering.  
- **Challenges:** Heavy class imbalance, many identities with very few samples, unconstrained "in the wild" conditions (pose, lighting, occlusion).

**Preprocessing pipeline includes:**
- Download and clean/split (train/val/test) with fixed seed.
- Face alignment / resize to 160×160 PNG.
- Consistent class indexing across splits (empty class folders preserved to avoid shifting label indices).
- Data augmentation for CNN: random horizontal flip, small rotation, color jitter; normalized to mean=0.5, std=0.5.

---

## 3) Models & Training

### Classical Models
- Input: fixed MobileNetV2 embeddings (no further fine-tuning).
- Scaling: `StandardScaler` applied before SGD classifiers.
- Hyperparameter tuning: small manual/randomized search over `alpha`, `eta0`, learning rate schedule for SGD variants; limited search for KNN (k and metric).

### CNN Fine-tuning
- Backbone: `torchvision.models.mobilenet_v2` with ImageNet weights.
- Head: dropout + linear layer adapted to number of classes.
- Only last N parameters are unfrozen (Optuna choice) to control capacity and overfitting.
- Optimizer: `AdamW`.
- Validation monitoring with early stopping; best model checkpointed.

---

## 4) Results Summary (as observed)

| Model                     | Validation Acc. | Test Acc.       | Notes |
|--------------------------|-----------------|-----------------|-------|
| **MobileNetV2 (fine-tuned)** | **0.840**       | ~0.73–0.79      | Optuna-tuned over LR, dropout, weight decay, unfreeze depth; early stopping. |
| SGD-Logistic (embeddings) | 0.413           | —               | Small manual/random search; fast. |
| SGD-SVM (embeddings)      | 0.410           | —               | Similar performance to logistic. |
| K-NN (embeddings)         | 0.247           | —               | Simple non-parametric baseline. |

*Interpretation:* Classical baselines give quick, lightweight signals but are substantially outperformed by the fine-tuned CNN, which adapts features to the specific identity distribution despite class scarcity.

---

## 5) Repository Structure

