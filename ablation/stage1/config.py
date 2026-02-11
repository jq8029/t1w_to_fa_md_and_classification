import torch

# Path Options
train_dir = "/project/hcp/data/ablation/train"
test_dir = "/project/hcp/data/ablation/test"

# File OPtions
T1W_FILE_NAME = "T1w_acpc_dc_restore_1.25.nii.gz"
FA_FILE_NAME = "dtifit_FA.nii.gz"
MD_FILE_NAME = "dtifit_MD.nii.gz"
MASK_FILE_NAME = "nodif_brain_mask.nii.gz"
BEST_MODEL = "/project/hcp/model/ablation_best_model.dt"

# Config Options
LR = 1e-4
BASE_CHANNELS = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

