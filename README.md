# Multi-modal MRI-Based Alzheimer's Disease Diagnosis with Transformer-based Image Synthesis and Transfer Learning

This repository implements a transformer-based image synthesis framework for predicting white-matter microstructural measures—**Fractional Anisotropy (FA)** and **Mean Diffusivity (MD)**—from **T1-weighted MRI (T1w)**, and leveraging **multimodal MRI data** for **binary and three-class classification** tasks.

## 1. Introduction
Diffusion-derived metrics such as FA and MD are sensitive indicators of white-matter integrity but require diffusion tensor imaging (DTI), which is not always available in clinical settings. This work demonstrates that high-quality diffusion microstructural information can be inferred from routinely acquired T1-weighted MRI, effectively transferring the benefits of multi-modality imaging to settings where diffusion data are unavailable. .

The proposed framework consists of two sequential stages:

- **Stage 1**: Diffusion metric prediction  
  Fractional Anisotropy (FA) and Mean Diffusivity (MD) are predicted from
  T1-weighted MRI using a regression network.

- **Stage 2**: Multi-modal classification  
  T1-weighted MRI together with the predicted FA and MD are fused for binary and three-class classification tasks.

## 2. Stage 1 : Synthesis Model
- T1w → FA
- T1w → MD

### 2.1 Input
- T1-weighted MRI (T1w) 

### 2.2 Output
- Predicted FA, MD

### 2.3 Model
- 3D TransUNet
- Convolutional layers: 3D convolution + BatchNorm + ReLU
- Base Chanel 32

## 2.4. Dataset and Preprocessing
- All images are spatially registered to a common template
- Intensity normalization using Z-score normalization
- Data split at the subject level:
  - Training / Validation / Testing

## 2.5 Implementation Details
- Optimizer: Adam
- Loss Function: L1
- Initial learning rate: 1e-4
- Batch size: 10
- Patch SiZe: 64 x 64 x 64
- Early Stop: Patience = 10
- Training performed on NVIDIA RTX-3070 GPUs

### 2.6 Evaluation Metrics
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Pearson correlation coefficient
  - Structural Similarity Index (SSIM)

## 3. Stage 2 : Prediction Model
- Binary classification
- Three-class classification

### 3.1 Input Modalities
- T1-weighted MRI (T1w)
- Predicted FA
- Predicted MD

### 3.2 Output
- Class probabilities
- Final predicted labels

### 3.3 Model
- Standard 2D Convolutional Neural Network (CNN)

## 3.4. Dataset and Preprocessing
- All images are spatially registered to a common template
- Intensity normalization using Z-score normalization
- Data split performed at the subject level:
  - Training / Validation / Testing

## 3.5 Implementation Details
- 2D Slice Images: 30 slices / subject
- Optimizer: Adam
- Loss Function: 
  - Binary: BCEWithLogitsLoss
  - Three-class: CrossEntropyLoss
- Initial learning rate
  - Binary: 3e-4
  - Three-class: 5e-4
- Batch size: 30
- Early Stop: Patience = 10
- Training performed on NVIDIA RTX-3070 GPUs

### 3.6 Evaluation Metrics
  - Accuracy – Measures the proportion of all predictions that are correct
  - Precision - The proportion of predicted positive cases that are actually positive
  - Recall – The proportion of actual positive cases that are correctly identified
  - F1 Score – measures the harmonic mean of precision and recall and effectively reflects the balance between the two values.
  - Subject-Level AUC – measures the model’s ability to distinguish between 

## 4. Ablation Studies
Extensive ablation studies are conducted to analyze the contribution of each component in Stage 1 and Stage 2 of the proposed framework.

## 5. Experiments
The experimental results demonstrate that:
- FA and MD can be reliably approximated from T1w MRI
- Multimodal learning consistently improves both binary and multi-class classification performance
- Predicted FA/MD features provide complementary information for classification tasks

## 6. Reproducibility
- Random seeds are fixed for all experiments
- Training and evaluation scripts are provided
- Configuration files define all hyperparameters

## 7. License
MIT

**Note: Due to data privacy regulations, raw imaging data are not publicly released.**

## 8. Citation
If you find this work useful, please cite:

```bibtex
@misc{arxiv2601_11614,
  title         = {Multi-modal MRI-Based Alzheimer's Disease Diagnosis with Transformer-based Image Synthesis and Transfer Learning},
  author        = {Jason Qiu},
  year          = {2026},
  eprint        = {2601.11614},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2601.11614}
}
```

## 9. Others 
### 9.1 Project Structure
```text

├── stage1_synthesis_model   # Stage 1 : implements the synthesis model for predicting FA and MD from T1w MRI 
├── stage2/                  # Stage 2: implements the multi-modal classification network 
├── ablation/                # Contains experiments evaluating the contribution of each component
│   ├── stage1/              # Ablation on Stage 1
│   └── stage2/              # Ablation on Stage 2  
├── scripts/                 # Contains preprocessing scripts for HCP and ADNI datasets.
└── README.md
```

### 9.2 Data

This project uses data from the following public datasets:

- Human Connectome Project (HCP)
- Alzheimer's Disease Neuroimaging Initiative (ADNI)

Due to data usage agreements, the raw data cannot be redistributed.

### 9.3 Data Download

- HCP data can be accessed via https://db.humanconnectome.org
- ADNI data can be accessed via https://adni.loni.usc.edu

### 9.4 Preprocessing

- T1w images were preprocessed using FSL
- FA and MD maps were computed using FSL `dtifit` and `bet`
- Detailed preprocessing scripts are provided in `scripts/`