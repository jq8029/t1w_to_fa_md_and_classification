# Stage 1 : Synthesis Model
In Stage 1, a 3D TransUNet is trained to capture the long-range dependencies in the images, and to achieve high-fidelity FA and MD prediction from T1w input.

## How to Run

- **Training**  
python train.py

- **Test**  
python test.py

- **Inference**  
python inference.py

## Notes
Please update file and path options in 'config.py' before execution. 

The pretrained model weights (.pt) are provided separately due to file size constraints.
Please download the file from the link below and place it in this folder before execution.
https://drive.google.com/file/d/12t9td1OhX6CHC0bWNmU55a3DJV1ifCGE/view?usp=sharing

## Preparing Outputs for Stage 2
To use the predicted FA and MD results from Stage 1 as inputs for Stage 2, please follow the steps below:

To predict FA, MD for stage 2, please follow below steps:

**a. Ensure consistent file naming**  
The following file names must be consistent between Stage 1 and Stage 2 configuration files:

```text
PRED_FA_FILE_NAME  <=> FA_FILE_NAME
PRED_MD_FILE_NAME  <=> MD_FILE_NAME 
```

**b. Set the inference output directory**  
In the Stage 1 config.py, set ST1_INFR_DIR to point to the corresponding training directories of Stage 2 in turn:

```text
ST2_TRAIN_NC_DIR
ST2_TRAIN_MCI_DIR
ST2_TRAIN_AD_DIR 
```
These directories should contain the generated FA and MD predictions from Stage 1.

