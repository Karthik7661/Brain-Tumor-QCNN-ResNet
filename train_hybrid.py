
#TRAING FULL CODE

!pip install pennylane
!pip install --quiet gradio

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.models as models
import pennylane as qml
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# CONFIG
DATASET_PATH = "/kaggle/input/karthik-braintypesdata-mri/brain_Tumor_Types"
BATCH_SIZE = 16
EPOCHS = 25
LR = 2e-4
N_QUBITS = 4
SAVE_PATH = "hybrid_qcnn.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# DATA
train_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224,224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = ImageFolder(DATASET_PATH, transform=train_tf)
class_names = dataset.classes
num_classes = len(class_names)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
val_ds.dataset.transform = val_tf

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# QUANTUM LAYER
dev_q = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev_q, interface="torch")
def qnode(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

weight_shapes = {"weights": (3, N_QUBITS, 3)}
q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

class QuantumFiLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(N_QUBITS, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        B = x.size(0)
        pooled = F.adaptive_avg_pool2d(x, (1, N_QUBITS)).view(B, N_QUBITS)
        pooled = (pooled - pooled.min(dim=1, keepdim=True)[0])
        pooled = pooled / (pooled.max(dim=1, keepdim=True)[0] + 1e-9)
        pooled = pooled * np.pi

        q_out = q_layer(pooled)
        scale, shift = self.fc(q_out).tanh().unbind(dim=1)
        return x * (1 + scale.view(B,1,1,1)) + shift.view(B,1,1,1)

# HYBRID MODEL
class HybridQuantumCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.qfilm = QuantumFiLM()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.qfilm(x)
        x = self.backbone(x)
        return self.head(x)

model = HybridQuantumCNN().to(device)


# TRAINING SETUP
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# TRAIN LOOP
best_val = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item() * x.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(out.argmax(1).cpu().numpy())

    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save({
            "model_state": model.state_dict(),
            "class_names": class_names,
            "n_qubits": N_QUBITS
        }, SAVE_PATH)
        print("✔ Saved best model")

# FINAL REPORT

print("\nClassification Report")
print(classification_report(y_true, y_pred, target_names=class_names))
print("Confusion Matrix")
print(confusion_matrix(y_true, y_pred))
