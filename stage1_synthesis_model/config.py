import torch

# ---------------
# --- Stage 1 ---
#----------------

# File Options
T1W_FILE_NAME = "t1w.nii.gz"
FA_FILE_NAME = "dtifit_fa.nii.gz"
MD_FILE_NAME = "dtifit_md.nii.gz"
MASK_FILE_NAME = "brain_mask.nii.gz"
ST1_TRAIN_BEST_MODEL = "/project/hcp/model/train_best_model.dt"
PRED_FA_FILE_NAME = "Predict_FA.nii.gz"
PRED_MD_FILE_NAME = "Predict_MD.nii.gz"

# Path Options
ST1_TRAIN_DIR = "/prject/hcp/data/train"
ST1_TEST_DIR = "/project/hcp/data/test"
ST1_INFR_DIR = "/project/hcp/data/infr"

# Model
IN_CHANNELS = 1
OUT_CHANNELS = 2
BASE_CHANNELS = 32
NUM_WORKERS = 0
BLOCK_SIZE = 64

# Train
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ST1_BATCH_SIZE = 10
ST1_LR = 1e-4
ST1_EPOCHS = 200
ST1_PATIENCE = 10

# Test
ST1_BEST_MODEL_FILE="stage1_best_model.pt"


