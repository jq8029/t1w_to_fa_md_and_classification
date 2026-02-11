import os
import torch
import numpy as np
import utils as ut
import config as cf
from transunet import TransUNet

model = TransUNet(in_channels=cf.IN_CHANNELS, out_channels=cf.OUT_CHANNELS, base=cf.BASE_CHANNELS).to(cf.DEVICE)
state = torch.load(cf.ST1_BEST_MODEL_FILE, map_location="cpu")
model.load_state_dict(state, strict=False)
root_dir = cf.ST1_INFR_DIR


# loop all dirs under root dir
for subject_dir in sorted(d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))):
    
    t1w_img, _  = ut.load_image(root_dir, subject_dir, cf.T1W_FILE_NAME)
    mask_img, _ = ut.load_image(root_dir, subject_dir, cf.MASK_FILE_NAME)
    t1w_img = ut.zscore_normalize(t1w_img, mask_img)
    t1w_blocks, ind_block = ut.extract_blocks(t1w_img, mask_img)
    net_in = torch.from_numpy(t1w_blocks)

    with torch.no_grad():
        y_pred = model(net_in.to(device=cf.DEVICE,dtype=torch.float))
        y_pred = y_pred.cpu().numpy()
    
    y_pred = ut.assemble_blocks(y_pred, ind_block, mask_img)

    fa_out = y_pred[0,:,:,:]
    md_out = y_pred[1,:,:,:]

    # save predicted FA and MD images to the same subject folder
    fa_path = os.path.join(root_dir, subject_dir, cf.PRED_FA_FILE_NAME)
    ut.save_image(fa_out, fa_path)
    md_path = os.path.join(root_dir, subject_dir, cf.PRED_MD_FILE_NAME)
    ut.save_image(md_out, md_path)
