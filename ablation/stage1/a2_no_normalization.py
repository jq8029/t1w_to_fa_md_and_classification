import utils as ut
import config as cf
import torch.nn as nn
import torch.optim as optim
from transunet import TransUNet


# set fixed seed
ut.set_fixed_seed()


# Model
model = TransUNet(in_channels=1, out_channels=2, base=cf.BASE_CHANNELS).to(cf.DEVICE)

# Training
ut.train(model, cf.BEST_MODEL, cf.TEST_DIR, with_norm=False)

# Test
ut.test(model, cf.BEST_MODEL, cf.TEST_DIR, with_norm=False)
