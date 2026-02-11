import os
import torch
import numpy as np
import torch.nn as nn
import utils as ut
import config as cf
from transunet import TransUNet


model = TransUNet(in_channels=cf.IN_CHANNELS, out_channels=cf.OUT_CHANNELS, base=cf.BASE_CHANNELS).to(cf.DEVICE)
state = torch.load(cf.ST1_BEST_MODEL_FILE, map_location="cpu")
model.load_state_dict(state, strict=False)
criterion = nn.L1Loss()

fa_all_mse, fa_all_mae, fa_all_rmse, fa_all_r, fa_all_ssim = [], [], [], [], []
md_all_mse, md_all_mae, md_all_rmse, md_all_r, md_all_ssim = [], [], [], [], []

# loop all dirs under root dir
for subject_dir in sorted(d for d in os.listdir(cf.ST1_TEST_DIR)
        if os.path.isdir(os.path.join(cf.ST1_TEST_DIR, d))):

    t1w_img, _  = ut.load_image(cf.ST1_TEST_DIR, subject_dir, cf.T1W_FILE_NAME)
    fa_img, _   = ut.load_image(cf.ST1_TEST_DIR, subject_dir, cf.FA_FILE_NAME)
    md_img, _   = ut.load_image(cf.ST1_TEST_DIR, subject_dir, cf.MD_FILE_NAME, rescaling=True) 
    mask_img, _ = ut.load_image(cf.ST1_TEST_DIR, subject_dir, cf.MASK_FILE_NAME)

    # normlize t1w
    t1w_img = ut.zscore_normalize(t1w_img, mask_img)

    t1w_blocks, ind_block = ut.extract_blocks(t1w_img, mask_img)
    fa_blocks, _  = ut.extract_blocks(fa_img, mask_img)
    md_blocks, _  = ut.extract_blocks(md_img, mask_img)

    net_in = torch.from_numpy(t1w_blocks)
    output_gt = np.concatenate([fa_blocks,  md_blocks], axis = 1)
    y_gt = torch.from_numpy(output_gt)

    with torch.no_grad():
        y_pred = model(net_in.to(device=cf.DEVICE,dtype=torch.float))
        y_gt = y_gt.to(device=cf.DEVICE,dtype=torch.float)
        test_loss = criterion(y_pred, y_gt)

        y_pred = y_pred.cpu().numpy()

    y_pred = ut.assemble_blocks(y_pred, ind_block, mask_img)
    fa_pred_out = y_pred[0,:,:,:]
    md_pred_out = y_pred[1,:,:,:]
    fa_gt_out = fa_img[0,:,:,:]
    md_gt_out = md_img[0,:,:,:]

    mse = np.mean((fa_pred_out - fa_gt_out)**2)
    mae = np.mean(np.abs(md_pred_out - md_gt_out))
    rmse = np.sqrt(mse)
    r = np.corrcoef(fa_pred_out.flatten(), fa_gt_out.flatten())[0,1]
    ssim_mean = ut.compute_ssim_mean(fa_pred_out, fa_gt_out)

    print("FA : MSE:", mse, "MAE:", mae, "RMSE:", rmse, "Pearson r:", r, "SSIM:", ssim_mean)

    fa_all_mse.append(mse)
    fa_all_mae.append(mae)
    fa_all_rmse.append(rmse)
    fa_all_r.append(r)
    fa_all_ssim.append(ssim_mean)

    mse = np.mean((md_pred_out - md_gt_out)**2)
    mae = np.mean(np.abs(md_pred_out - md_gt_out))
    rmse = np.sqrt(mse)
    r = np.corrcoef(md_pred_out.flatten(), md_gt_out.flatten())[0,1]
    ssim_mean = ut.compute_ssim_mean(md_pred_out, md_gt_out)

    print("MD : MSE:", mse, "MAE:", mae, "RMSE:", rmse, "Pearson r:", r, "SSIM:", ssim_mean)

    md_all_mse.append(mse)
    md_all_mae.append(mae)
    md_all_rmse.append(rmse)
    md_all_r.append(r)
    md_all_ssim.append(ssim_mean)

mse = np.mean(fa_all_mse)
mae = np.mean(fa_all_mae)
rmse = np.mean(fa_all_rmse)
r = np.mean(fa_all_r)
ssim_mean = np.mean(fa_all_ssim)
print("Final FA - MSE:", mse, "MAE:", mae, "RMSE:", rmse, "Pearson r:", r, "SSIM:", ssim_mean)

mse = np.mean(md_all_mse)
mae = np.mean(md_all_mae)
rmse = np.mean(md_all_rmse)
r = np.mean(md_all_r)
ssim_mean = np.mean(md_all_ssim)
print("Final MD - MSE:", mse, "MAE:", mae, "RMSE:", rmse, "Pearson r:", r, "SSIM:", ssim_mean)