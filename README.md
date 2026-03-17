# 3D_Unet_from_scratch
3D U‑Net implementation in PyTorch from scratch using PyTorch to train a neural network for MRI images


# 🧠 3D U‑Net — Volumetric Segmentation from Scratch
This repository contains a from‑scratch implementation of the 3D U‑Net architecture, a convolutional neural network designed for volumetric (3D) segmentation. It is widely used in medical imaging tasks such as MRI and CT scan analysis, where understanding spatial relationships across all three dimensions is essential.
The project includes the full model architecture, training pipeline, configuration system, and example trained weights.

# 📁 Project Structure
```
3D_Unet_from_scratch/
│
├── assets/                         # Images, plots, and visual resources
├── carvana_dataset.py              # Example dataset class (customizable)
├── config.py                       # Hyperparameter and path configuration
├── entrenamiento_stats.png         # Training curves (loss/metrics)
├── model.pth                       # Example trained model
├── unet2D_parts.py                 # 2D U-Net components (if needed)
├── unet3D_parts.py                 # 3D U-Net building blocks
├── unet_3D_main.py                 # Main training script
├── unet3d_brain_model.pth          # Pretrained 3D brain segmentation model
└── README.md                       # This file
```
