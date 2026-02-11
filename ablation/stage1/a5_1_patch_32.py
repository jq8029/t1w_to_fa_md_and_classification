import utils as ut
import config as cf
from transunet import TransUNet

# set fixed seed
ut.set_fixed_seed()

# Model
model = TransUNet(in_channels=1, out_channels=2, base=cf.BASE_CHANNELS).to(cf.DEVICE)

# Training - Patch Size = 32
ut.train(model, cf.BEST_MODEL, cf.TEST_DIR, block_size=32)

# Test - Patch Size = 32
ut.test(model, cf.BEST_MODEL, cf.TEST_DIR, block_size=32)