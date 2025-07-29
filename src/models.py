# src/models.py
import torch.nn as nn
from torchvision import models

# -------- Custom small CNN --------
class TinyCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                               # 80×80
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                               # 40×40
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                               # 20×20
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)                        # 1×1×256
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# -------- Transfer‑learning helper --------
def mobilenet_v2_finetune(num_classes: int, unfreeze_from=100):
    net = models.mobilenet_v2(weights="IMAGENET1K_V1")
    # freeze all
    for p in net.parameters(): p.requires_grad_(False)
    # unfreeze last `unfreeze_from` parameters
    for p in list(net.parameters())[-unfreeze_from:]:
        p.requires_grad_(True)

    net.classifier[1] = nn.Linear(net.last_channel, num_classes)
    return net
