!pip install pennylane
!pip install --quiet gradio# hybrid_quantum_before_cnn.py
# Hybrid model: Quantum (FiLM-style modulation) BEFORE CNN backbone (ResNet18).
# Purpose: classify 4 classes: (3 tumor types) + (no_tumor)
# Author: generated for you

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import pennylane as qml
import numpy as np
import time

# ---------------------------
# 1) USER: set dataset path
# ---------------------------
dataset_path = "/kaggle/input/karthik-braintypesdata-mri/brain_Tumor_Types"  # change to your path
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

# ---------------------------
# 2) Transforms & dataloaders
# ---------------------------
# Use grayscale -> keep single channel
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # minor jitter (works on single channel)
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_dataset = ImageFolder(root=dataset_path, transform=train_transform)
class_names = full_dataset.classes
print("Classes found:", class_names)

# split train / val (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Ensure val uses val_transform (replace transform of subset)
val_dataset.dataset.transform = val_transform

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# ---------------------------
# 3) Quantum module BEFORE CNN
#    - We use a small quantum circuit that consumes a compact summary of the
#      image (average-pooled patches) and returns a small vector.
#    - That vector is used to compute per-image FiLM parameters (scale & shift)
#      that modulate the input image tensor before feeding the CNN backbone.
# ---------------------------

n_qubits = 4  # small; tune if desired
dev = qml.device("default.qubit", wires=n_qubits)

# Quantum circuit expects an input of length n_qubits and returns n_qubits expectation values
@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    # inputs: shape (n_qubits,) values in [0, pi] ideally
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# weight shapes for TorchLayer
weight_shapes = {"weights": (3, n_qubits, 3)}  # 3 layers of strongly entangling (tunable)
q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

class QuantumFiLM(nn.Module):
    """
    Quantum-FiLM: produce per-image scale & shift from quantum circuit outputs,
    then apply to input image (batch,1,H,W) as: x = x*(1+scale) + shift
    where scale and shift are broadcast per-image.
    """
    def __init__(self, n_qubits, out_features=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layer = q_layer  # TorchLayer mapping -> R^{n_qubits}
        # Map quantum outputs to (scale, shift) scalars (per channel if needed).
        # For grayscale we output 2 numbers: scale and shift.
        self.post = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, out_features)  # out_features=2 => [scale, shift]
        )

    def forward(self, x):
        # x: (B, C=1, H, W), values ~ normalized [-1,1] by transforms.Normalize
        B = x.shape[0]
        # produce compact classical inputs for qnode:
        # Use adaptive average pooling to create n_qubits values per image
        pooled = F.adaptive_avg_pool2d(x, (1, self.n_qubits)).view(B, self.n_qubits)  # shape (B, n_qubits)
        # normalize pooled to [0, pi] to match AngleEmbedding scale
        pooled_np = pooled  # torch tensor
        pooled_norm = (pooled_np - pooled_np.min(dim=1, keepdim=True)[0])  # avoid negative
        # avoid zero division
        denom = pooled_np.max(dim=1, keepdim=True)[0] - pooled_np.min(dim=1, keepdim=True)[0] + 1e-9
        pooled_scaled = pooled_norm / denom
        inputs_q = pooled_scaled * np.pi  # range [0, pi]
        # Now pass each row to quantum layer independently (TorchLayer handles batch)
        q_out = self.q_layer(inputs_q)  # shape (B, n_qubits)
        # map to scale and shift
        params = self.post(q_out)  # (B, out_features)
        scale = torch.tanh(params[:, 0]).view(B, 1, 1, 1)  # bounded scale in (-1,1)
        shift = torch.tanh(params[:, 1]).view(B, 1, 1, 1)  # bounded shift in (-1,1)
        # apply FiLM modulation
        x_mod = x * (1.0 + scale) + shift
        return x_mod, q_out  # return modulated image; also q_out for possible diagnostics

# ---------------------------
# 4) Hybrid model: QuantumFiLM -> CNN (ResNet18) -> Classifier
# ---------------------------
class HybridQuantumBeforeCNN(nn.Module):
    def __init__(self, num_classes, n_qubits=4, pretrained=True):
        super().__init__()
        self.quantum_film = QuantumFiLM(n_qubits=n_qubits, out_features=2)

        # Use ResNet18 backbone (lightweight), adapt for 1-channel input
        self.backbone = models.resnet18(pretrained=pretrained)
        # Change first conv to accept 1 channel (grayscale)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # optionally freeze some layers to speed up training / regularize
        # for name, param in self.backbone.named_parameters():
        #     if "layer4" not in name:  # keep last block trainable
        #         param.requires_grad = False

        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove original classifier

        # a small classifier head after backbone
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Quantum modulation before CNN
        x_mod, q_out = self.quantum_film(x)  # x_mod same shape as x
        # pass through CNN backbone
        feats = self.backbone(x_mod)  # (B, feat_dim)
        logits = self.classifier(feats)
        return logits, q_out

# ---------------------------
# 5) Setup model, device, optimizer, loss
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = HybridQuantumBeforeCNN(num_classes=len(class_names), n_qubits=n_qubits, pretrained=True)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ---------------------------
# 6) Training loop (monitor val metrics)
# ---------------------------
epochs = 25  # start here, you can increase for better performance
best_val_loss = float('inf')
save_path = "best_hybrid_quantum_before_cnn.pth"

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    t0 = time.time()
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, q_out = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)

    # validation
    model.eval()
    val_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits, q_out = model(imgs)
            loss = criterion(logits, labels)
            val_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    epoch_val_loss = val_loss / len(val_loader.dataset)
    scheduler.step(epoch_val_loss)

    elapsed = time.time() - t0
    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {epoch_train_loss:.4f} Val Loss: {epoch_val_loss:.4f} Time: {elapsed:.1f}s")

    # save best
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "class_names": class_names
        }, save_path)
        print(f"Saved best model (val_loss={best_val_loss:.4f})")

# ---------------------------
# 7) Final evaluation & report
# ---------------------------
# load best
ckpt = torch.load(save_path, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

y_true = []
y_pred = []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits, q_out = model(imgs)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ---------------------------
# 8) Optional: Gradio predict function (if you want a demo)
# ---------------------------
def predict_gradio(img_pil):
    # img_pil: PIL.Image
    img = img_pil.convert("L")  # grayscale
    img_t = val_transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits, _ = model(img_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        top_idx = np.argmax(probs)
        return {class_names[i]: float(probs[i]) for i in range(len(class_names))}

# Save the model checkpoint and provide the predict_gradio for local demo usage (not launching here).
print("Script finished. Best model saved to:", save_path)
