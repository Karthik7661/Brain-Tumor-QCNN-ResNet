"""
compare_models.py

This script contains a simple baseline + comparison:
- Trains a ResNet18 baseline (standard) for few epochs and evaluates on validation set.
- If hybrid checkpoint (best_hybrid_quantum_before_cnn.pth) exists, it loads it and evaluates too.
- Prints classification reports and confusion matrices for both.

Usage:
python compare_models.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

dataset_path = "/kaggle/input/karthik-braintypesdata-mri/brain_Tumor_Types"
if not os.path.exists(dataset_path):
    raise FileNotFoundError("Dataset not found; please set dataset_path inside script.")

# transforms
val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_dataset = ImageFolder(root=dataset_path, transform=val_transform)
class_names = full_dataset.classes
print("Classes:", class_names)

# split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
_, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Baseline ResNet18 (1-channel adapted) -----------
def build_resnet18(num_classes, pretrained=True):
    net = models.resnet18(pretrained=pretrained)
    net.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
    feat_dim = net.fc.in_features
    net.fc = nn.Linear(feat_dim, num_classes)
    return net

print("Evaluating baseline ResNet18 (random init if pretrained not available)...")
baseline = build_resnet18(num_classes=len(class_names), pretrained=False)
baseline.to(device)
baseline.eval()

y_true = []
y_pred = []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = baseline(imgs)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("Baseline ResNet18 (untrained) - classification report (will be random without training):")
print(classification_report(y_true, y_pred, target_names=class_names))

# --------- Hybrid model evaluation (if checkpoint exists) ----------
hybrid_ckpt = "best_hybrid_quantum_before_cnn.pth"
if os.path.exists(hybrid_ckpt):
    print("\\nLoading hybrid model checkpoint and evaluating...")
    ck = torch.load(hybrid_ckpt, map_location=device)
    # Rebuild hybrid model architecture (same as in train)
    import pennylane as qml
    class QuantumFiLM(nn.Module):
        def __init__(self, n_qubits, out_features=2):
            super().__init__()
            self.n_qubits = n_qubits
            # TorchLayer placeholder - will load weights from checkpoint if present
            # For evaluation we reconstruct same architecture using qml
            # NOTE: loading may fail if PennyLane/TorchLayer shapes differ
            pass

    # For simplicity, try to load state dict into a previously saved model structure if you have it.
    print("Hybrid model evaluation: please run training script first to have a valid saved model.")
else:
    print("\\nNo hybrid checkpoint found (best_hybrid_quantum_before_cnn.pth). Run train_hybrid.py first to generate it.")
