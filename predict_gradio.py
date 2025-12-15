#PREDECTION FULL CODE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import pennylane as qml
import numpy as np
import gradio as gr
from PIL import Image

# LOAD CHECKPOINT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load("hybrid_qcnn.pth", map_location=device)
class_names = ckpt["class_names"]
N_QUBITS = ckpt["n_qubits"]

# QUANTUM DEFINITIONS
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
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(512, len(class_names))

    def forward(self, x):
        x = self.qfilm(x)
        x = self.backbone(x)
        return self.head(x)

# LOAD MODEL
model = HybridQuantumCNN().to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()


# INFERENCE TRANSFORM (LOCKED)
infer_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# PREDICTION FUNCTION
def predict_mri(img):
    img = infer_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(img), dim=1)[0].cpu().numpy()
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}

# GRADIO UI
gr.Interface(
    fn=predict_mri,
    inputs=gr.Image(type="pil", label="Upload MRI"),
    outputs=gr.Label(num_top_classes=len(class_names)),
    title="Hybrid Quantum-CNN Brain Tumor Classifier",
    description="Upload an MRI scan to classify tumor type or no tumor."
).launch(share=True)
