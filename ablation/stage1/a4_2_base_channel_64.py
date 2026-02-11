import utils as ut
import config as cf
from transunet import TransUNet

# set fixed seed
ut.set_fixed_seed()

# Model - Base Channel = 64
model = TransUNet(in_channels=1, out_channels=2, base=64).to(cf.DEVICE)


# Training
ut.train(model, cf.BEST_MODEL, cf.TEST_DIR)

# Test
ut.test(model, cf.BEST_MODEL, cf.TEST_DIR)