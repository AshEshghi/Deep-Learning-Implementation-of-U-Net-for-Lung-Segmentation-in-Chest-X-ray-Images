# Deep-Learning-Implementation-of-U-Net-for-Lung-Segmentation-in-Chest-X-ray-Images
This repository implements a U-Net deep learning model for binary lung segmentation on chest X-ray images. The project reproduces a standard medical image segmentation pipeline: preprocessing, training/validation/testing, metric reporting, and qualitative visualization.

---

## What is U-Net?
**U-Net** is a convolutional neural network architecture designed for **biomedical image segmentation**, introduced in 2015 by **Ronneberger, Fischer, and Brox**. Unlike classification models, U-Net performs **pixel-level prediction**, assigning a label to *every pixel* in an image.
The architecture is “U-shaped”:
- An **encoder (contracting path)** captures semantic features through downsampling.
- A **decoder (expanding path)** restores spatial resolution through upsampling.
- **Skip connections** link encoder and decoder layers at matching resolutions, allowing the decoder to recover fine spatial detail while using high-level context—critical for precise medical segmentation.
---

## Original Source and Credit
- This project is based on the seminal paper:

> **Ronneberger, O., Fischer, P., & Brox, T. (2015).**  
> *U-Net: Convolutional Networks for Biomedical Image Segmentation.* (MICCAI)
>
> **link:** https://arxiv.org/abs/1505.04597

All credit for the **U-Net architecture concept** belongs to the original authors. This repository is an **independent educational implementation** of the idea, adapted to lung segmentation on chest X-rays.

---

## What I Implemented

This repository contains my own implementation of:
- A **U-Net model** in PyTorch (encoder, decoder, skip connections)
- A **paired dataset loader** (X-ray image + lung mask)
- Train/validation/test split
- **Loss function:** BCEWithLogits + Dice (combined)
- **Metrics:** Dice coefficient and IoU
- Visualization utilities:
  - predicted masks
  - red overlay over the original X-ray
  - training curves (loss & dice)

> Note: While U-Net is a known architecture, the **code, training configuration, evaluation, and visual outputs** here were implemented for this project.

---

## Dataset

- **Type:** Chest X-ray images + corresponding binary lung masks  
- **Source:** Kaggle (excluded from this repo due to size ~4GB)  
- **How to use:** Download and place it under the `data/` directory following the folder structure used in the notebook.
- **link:** https://www.kaggle.com/datasets/iamtapendu/chest-x-ray-lungs-segmentation?resource=download

- ---

## Method

### Preprocessing
- Grayscale images (1 channel)
- Resize to **128×128** (hardware-safe)
- Convert to tensor in [0,1]
- Masks binarized to {0,1}

### Model
- **U-Net** encoder–decoder with skip connections  
- Output: logits → `sigmoid` → threshold at **0.5**

### Loss & Metrics
- **Loss:** 0.5 × BCEWithLogits + 0.5 × (1 − Dice)
- **Metrics:** Dice coefficient and IoU

---

## Training Setup

- Split: **80% train / 10% validation / 10% test**
- Batch size: **2** (chosen to avoid laptop crashes)
- Epochs: **3**
- Optimizer: Adam (lr=1e-3)
- Best model saved based on **highest validation Dice**

---

## Results

### Quantitative
The validation Dice improved over epochs and the best model was saved as:

- `results/best_unet.pth`

(Your exact numbers may vary slightly due to randomness.)

### Qualitative
Prediction examples are saved as:
- `results/preds/` (raw outputs)
- `results/montage/` (recommended for viewing)

Each montage shows: **X-ray | Ground Truth | Prediction | Overlay**.

Training curves are saved as:
- `results/loss_curves.png`
- `results/dice_curves.png`

---

## Repository Structure

```text
├── UNet_Lung_Segmentation.ipynb
├── README.md
├── results/
│   ├── best_unet.pth
│   ├── loss_curves.png
│   ├── dice_curves.png
│   ├── preds/
│   └── montage/
└── data/   (not uploaded)
