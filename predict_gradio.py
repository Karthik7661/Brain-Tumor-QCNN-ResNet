import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import pennylane as qml
import numpy as np
import torch.nn as nn
import torchvision.models as models
import gradio as gr
import os

# Load classes and define transforms (update path as needed)
dataset_path = "/kaggle/input/karthik-braintypesdata-mri/brain_Tumor_Types"
if not os.path.exists(dataset_path):
    print("Dataset path not found; prediction still works with uploaded images.")

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define small wrappers for model (must match train_hybrid.py)
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (3, n_qubits, 3)}
q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

class QuantumFiLM(nn.Module):
    def __init__(self, n_qubits, out_features=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layer = q_layer
        self.post = nn.Sequential(nn.Linear(n_qubits,16), nn.ReLU(), nn.Linear(16,out_features))

    def forward(self, x):
        B = x.shape[0]
        pooled = torch.nn.functional.adaptive_avg_pool2d(x,(1,self.n_qubits)).view(B,self.n_qubits)
        pooled_np = pooled
        pooled_norm = (pooled_np - pooled_np.min(dim=1,keepdim=True)[0])
        denom = pooled_np.max(dim=1,keepdim=True)[0] - pooled_np.min(dim=1,keepdim=True)[0] + 1e-9
        pooled_scaled = pooled_norm / denom
        inputs_q = pooled_scaled * np.pi
        q_out = self.q_layer(inputs_q)
        params = self.post(q_out)
        scale = torch.tanh(params[:,0]).view(B,1,1,1)
        shift = torch.tanh(params[:,1]).view(B,1,1,1)
        x_mod = x * (1.0 + scale) + shift
        return x_mod, q_out

class HybridQuantumBeforeCNN(nn.Module):
    def __init__(self, num_classes, n_qubits=4, pretrained=True):
        super().__init__()
        self.quantum_film = QuantumFiLM(n_qubits=n_qubits, out_features=2)
        self.backbone = models.resnet18(pretrained=pretrained)
        self.backbone.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(nn.Linear(feat_dim,256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256,4))

    def forward(self,x):
        x_mod, q_out = self.quantum_film(x)
        feats = self.backbone(x_mod)
        logits = self.classifier(feats)
        return logits, q_out

# load model checkpoint if present
checkpoint_path = "best_hybrid_quantum_before_cnn.pth"
if os.path.exists(checkpoint_path):
    ck = torch.load(checkpoint_path, map_location="cpu")
    class_names = ck.get("class_names", ["class0","class1","class2","class3"])
    model = HybridQuantumBeforeCNN(num_classes=len(class_names))
    model.load_state_dict(ck["model_state"])
    model.eval()
else:
    # fallback labels
    class_names = ["class0","class1","class2","class3"]
    model = HybridQuantumBeforeCNN(num_classes=4)
    model.eval()

def predict_image(img):
    import numpy as np
    img_pil = Image.fromarray(img).convert("L")
    img_tensor = val_transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(img_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}

# Gradio interface
import numpy as np
gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", image_mode="L"),
    outputs=gr.Label(num_top_classes=4),
    title="Brain Tumor QCNN (Hybrid ResNet)",
    description="Upload MRI image (grayscale) for classification"
).launch()
