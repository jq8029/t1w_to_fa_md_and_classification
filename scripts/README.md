# Scripts

This directory contains preprocessing scripts for HCP and ADNI datasets.

Each script is designed to be run independently and produces standardized T1w and diffusion-derived outputs used in downstream analyses.

## 1. HCP preprocessing

**Script:** `preprocess_hcp.sh`

This script processes data from the Human Connectome Project (HCP) and generates T1-weighted structural images and diffusion tensor metrics.

### Input
- All HCP subject directory should under data/ 

### Output
- `T1w_acpc_dc_restore_1.25.nii.gz`  # (resampled T1-weighted image at 1.25 mm resolution) 
- `dtifit_FA.nii.gz`  
- `dtifit_MD.nii.gz`
- `nodif_brain_mask.nii.gz`

### Usage
```bash
bash preprocess_hcp.sh
```

## 2. ADNI preprocessing

**Script:** `preprocess_adni.sh`

This script performs brain extraction on ADNI T1-weighted images.

### Input
- All ADNI subject directory should under data/ 
  - `data/ad`
  - `data/mci`
  - `data/nc`

### Output
- `T1w_brain.nii.gz`
- `T1w_brain_out_bet.nii.gz`

### Usage
```bash
bash preprocess_adni.sh
```

## 3. Notes

- Raw HCP and ADNI data are not distributed due to data usage agreements.
- Users must obtain appropriate access permissions from the original data repositories.
- Visual quality control of all outputs is strongly recommended.