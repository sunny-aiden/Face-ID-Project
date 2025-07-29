# src/data.py
from pathlib import Path
import numpy as np, tensorflow as tf, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMG_SIZE = (160,160)

# ----------   PyTorch loaders   ----------
def make_loaders(proc_root: Path, batch=32, num_workers=2):
    train_tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.RandomResizedCrop(160, scale=(0.95,1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_ds = datasets.ImageFolder(proc_root/"train", transform=train_tfms)
    val_ds   = datasets.ImageFolder(proc_root/"val",   transform=val_tfms)
    test_ds  = datasets.ImageFolder(proc_root/"test",  transform=val_tfms)

    kwargs = dict(batch_size=batch, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available())
    train_ld = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_ld   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_ld  = DataLoader(test_ds,  shuffle=False, **kwargs)
    return train_ld, val_ld, test_ld, train_ds.classes

# ----------   Bottleneck extractor   ----------
def extract_features(proc_root: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)
    base = tf.keras.applications.MobileNetV2(include_top=False,
              weights="imagenet", input_shape=(*IMG_SIZE,3), pooling="avg")

    def dump(split):
        ds = tf.keras.utils.image_dataset_from_directory(
            proc_root/split, image_size=IMG_SIZE, batch_size=64, shuffle=False)
        X, y = [], []
        for xb, yb in ds:
            feats = base(tf.cast(xb, tf.float32)/255.0, training=False).numpy()
            X.append(feats);  y.append(yb.numpy())
        np.save(out_dir/f"X_{split}.npy", np.vstack(X))
        np.save(out_dir/f"y_{split}.npy", np.concatenate(y))

    for s in ("train","val","test"): dump(s)
