<p align="center">
  <img src="assets/bannerdl.jpg" alt="Brain Tumour QCNN Banner" width="100%">
</p>

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Badge">
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-orange" alt="PyTorch Badge">
  <img src="https://img.shields.io/badge/PennyLane-QuantumML-purple" alt="PennyLane Badge">
  <img src="https://img.shields.io/badge/Gradio-UI-green" alt="Gradio Badge">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status Badge">
  <img src="https://img.shields.io/badge/License-None-lightgrey" alt="License Badge">

</p>

# Brain Tumor Classification using Hybrid QCNN with ResNet

Hybrid quantum–classical model combining a small **Quantum FiLM** modulation (PennyLane) applied *before* a ResNet-50 backbone. The model classifies MRI images into multiple brain tumour categories (glioma, meningioma, pituitary) plus `no_tumor`. This repo includes training, inference (Gradio), and baseline comparison scripts.

---

##  Features

- Quantum FiLM modulation (per-image scale & shift) computed by a 4-qubit PennyLane circuit  
- ResNet-18 backbone adapted for grayscale medical images  
- End-to-end training & validation pipeline with checkpoint saving  
- Gradio-based image upload UI for quick demo / inference  
- Comparison script to evaluate a standard ResNet baseline  
- Designed for reproducible experiments and easy extension

---

##  Tech Stack

- Python 3.8+  
- PyTorch & torchvision (modeling & training)  
- PennyLane (quantum circuit + TorchLayer)  
- NumPy, PIL (data handling)  
- scikit-learn (metrics)  
- Gradio (prediction UI)  

### Utilities
- `torch.optim` optimizers & LR schedulers  
- `torchvision.transforms` for augmentation & normalization  
- `DataLoader`, `random_split` for reproducible train/val splits

---

##  Dataset

Kaggle dataset by the author:  
https://www.kaggle.com/datasets/skarthik112/karthik-braintypesdata-mri

Folder structure expected:
brain_Tumor_Types/
├─ glioma/
├─ meningioma/
├─ pituitary/
└─ no_tumor/

## 🧠 Sample Results (2×2)

<table>
  <tr>
    <td align="center"><img src="assets/result_glioma.png" width="300"><br>Glioma</td>
    <td align="center"><img src="assets/result_meningioma.png" width="300"><br>Meningioma</td>
  </tr>
  <tr>
    <td align="center"><img src="assets/result_pituitary.png" width="300"><br>Pituitary</td>
    <td align="center"><img src="assets/result_no_tumor.png" width="300"><br>No Tumor</td>
  </tr>
</table>
