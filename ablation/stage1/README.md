# Ablation - Stage 1
This directory contains ablation experiments based on the Stage 1 implementation. The code is a standalone variant of `stage1/`. 

Please update file and path options in 'config.py' before execution.

## How to Run
The naming and number of the ablation settings correspond exactly to the Stage 1 ablation experiments reported in the paper.

- **Table 2. Proposed model**  
python a0_proposed_model.py

- **Table 2. Transformer removed**  
python a1_transformer_removed.py

- **Table 2. No normalization**  
python a2_no_normalization.py

- **Table 2. L2 loss**  
python a3_l2_loss.py

- **Table 14. Base Channel (16)**  
python a4_1_base_channel_16.py

- **Table 14. Base Channel (64)**  
python a4_1_base_channel_64.py

- **Table 14. Patch Size 32**  
python a5_1_patch_32.py
