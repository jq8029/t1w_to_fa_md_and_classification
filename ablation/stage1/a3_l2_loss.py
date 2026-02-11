import utils as ut
import torch.nn as nn
import config as cf
import torch.optim as optim
from transunet import TransUNet

# set fixed seed
ut.set_fixed_seed()

input_train, input_val, output_train, output_val = ut.load_3D_images(cf.TRAIN_DIR)
train_loader, val_loader = ut.make_loaders(input_train, output_train, input_val, output_val)

# Model
model = TransUNet(in_channels=1, out_channels=2, base=cf.BASE_CHANNELS).to(cf.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=cf.LR)
criterion = nn.MSELoss()

# Training - Use L2 Loss
ut.train(model, cf.BEST_MODEL, cf.TRAIN_DIR, l1_loss=False)

# Test
ut.test(model, cf.BEST_MODEL, cf.TEST_DIR)