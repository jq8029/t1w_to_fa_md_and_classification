import torch

# ---------------
# --- Stage 2 ---
#----------------

# File Options
T1W_FILE_NAME         = "T1w_brain.nii.gz"
FA_FILE_NAME          = "Predict_FA.nii.gz"
MD_FILE_NAME          = "Predict_MD.nii.gz"
MASK_FILE_NAME        = "T1w_brain_out_bet.nii.gz"
ST2_3CLASS_BEST_MODEL = "/project/adni/model/three_class_best_model.dt"
ST2_BINARY_BEST_MODEL = "/project/adni/model/binary_best_model.dt"

# Path Options
ST2_TRAIN_NC_DIR  = "/project/adni/data/all/train/nc"
ST2_TRAIN_MCI_DIR = "/project/adni/data/all/train/mci"
ST2_TRAIN_AD_DIR  = "/project/adni/data/all/train/ad"
ST2_VAL_MCI_DIR   = "/project/adni/data/all/val/nc"
ST2_VAL_NC_DIR    = "/project/adni/data/all/val/mci"
ST2_VAL_AD_DIR    = "/project/adni/data/all/val/ad"
ST2_TEST_MCI_DIR  = "/project/adni/data/all/test/nc"
ST2_TEST_NC_DIR   = "/project/adni/data/all/test/mci"
ST2_TEST_AD_DIR   = "/project/adni/data/all/test/ad"

# Model
ST2_IN_CHANNELS = 1
ST2_NUM_CLASS_3CLASS = 3
ST2_NUM_BINARY = 2


# Train
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ST2_3CLASS_BATCH_SIZE = 30
ST2_3CLASS_LR = 5e-4
ST2_3CLASS_EPOCHS = 100
ST2_3CLASS_PATIENCE = 10
ST2_3CLASS_AD_POSSIBLE_THRESHOLD = 0.45
ST2_3CLASS_AD_SUBJECT_THRESHOLD = 8 
ST2_3CLASS_MCI_POSSIBLE_THRESHOLD = 0.25
ST2_3CLASS_MCI_SUBJECT_THRESHOLD = 26

ST2_BINARY_BATCH_SIZE = 30
ST2_BINARY_LR = 3e-4
ST2_BINARY_EPOCHS = 100
ST2_BINARY_PATIENCE = 10
ST2_BINARY_POSSIBLE_THRESHOLD = 0.25
ST2_BINARY_SUBJECT_THRESHOLD = 10  

