import os
import time
import torch
import random
import config as cf
import numpy as np
import torch.nn as nn
import nibabel as nib
import torch.optim as optim
import torch.nn.functional as F
from dataset import Dataset
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

def extract_blocks(data, mask, size=64):
    """
    data: (C, D, H, W)
    mask: (C, D, H, W)
    size: block size
    return:
        blocks: (N, C, size, size, size)
        ind_block: (N, 6)
    """
    _, D, H, W = data.shape

    # bbox
    mask_proc = mask.any(axis=0)
    xind, yind, zind = np.nonzero(mask_proc)
    xmin, xmax = xind.min(), xind.max()
    ymin, ymax = yind.min(), yind.max()
    zmin, zmax = zind.min(), zind.max()

    # main grid
    x_starts = np.linspace(xmin, xmax - size + 1, 3, dtype=int)
    y_starts = np.linspace(ymin, ymax - size + 1, 3, dtype=int)
    z_starts = np.linspace(zmin, zmax - size + 1, 3, dtype=int)

    blocks = []
    ind_block = []

    # main grid
    for xi in x_starts:
        for yi in y_starts:
            for zi in z_starts:
                ind_block.append([xi, xi+size-1, yi, yi+size-1, zi, zi+size-1])
                blocks.append(data[:, xi:xi+size, yi:yi+size, zi:zi+size])

    # shift one grid in D if needed
    offset = size // 2
    x_offset = [x_starts[0]]
    z_offset = np.linspace(zmin + offset, zmax - size + 1, 3, dtype=int)

    for xi in x_offset:
        for yi in y_starts:
            for zi in z_offset:
                ind_block.append([xi, xi+size-1, yi, yi+size-1, zi, zi+size-1])
                blocks.append(data[:, xi:xi+size, yi:yi+size, zi:zi+size])

    # align last block
    ind_block[-1] = [D-size, D-1, H-size, H-1, W-size, W-1]
    ind_block = np.array(ind_block, dtype=int)
    blocks[-1] = data[:, -size:, -size:, -size:]
    blocks = np.stack(blocks, axis=0)  # (N, C, size, size, size)

    return blocks, ind_block


def assemble_blocks(blocks, ind_block, mask):
    """
    blocks: (N, C, d, h, w)
    ind_block: (N, 6)
    mask: (C, D, H, W)
    return: (C, D, H, W)
    """

    N, C, d, h, w = blocks.shape

    # infer volume size from ind_block
    D = ind_block[:, 1].max() + 1
    H = ind_block[:, 3].max() + 1
    W = ind_block[:, 5].max() + 1

    vol = np.zeros((C, D, H, W), dtype=blocks.dtype)
    weight = np.zeros((D, H, W), dtype=np.float32)

    for i in range(N):
        x0, x1, y0, y1, z0, z1 = ind_block[i]
        vol[:, x0:x1+1, y0:y1+1, z0:z1+1] += blocks[i]
        weight[x0:x1+1, y0:y1+1, z0:z1+1] += 1

    weight = np.maximum(weight, 1.0)
    vol /= weight[None, ...]

    mask = mask.any(axis=0)
    vol *= mask[None, ...]

    return vol

def zscore_normalize(x, mask):
    """
    x:      (1, D, H, W)
    mask:   (1, D, H, W)
    return: (1, D, H, W)
    """
    mask = mask.astype(np.float32)

    num = mask.sum(axis=(1, 2, 3), keepdims=True)
    num = np.clip(num, a_min=1.0, a_max=None)

    mean = (x * mask).sum(axis=(1, 2, 3), keepdims=True) / num
    var = ((x - mean) ** 2 * mask).sum(axis=(1, 2, 3), keepdims=True) / num
    std = np.sqrt(var)
    std = np.clip(std, a_min=1e-6, a_max=None)

    x_norm = (x - mean) / std
    x_norm = x_norm * mask

    return x_norm


def load_image(root_dir, subject_dir, file_name, rescaling=False):
    full_path = os.path.join(root_dir,subject_dir, file_name)
    nii_file = nib.load(full_path)

    ras_idx = 0
    affine = nib.aff2axcodes(nii_file.affine)
    if (affine[0] == 'I'):
        ras_idx = 1  
    if (affine[0] == 'P'):
        ras_idx = 2

    img_data = nii_file.get_fdata(dtype=np.float32)  # (x,y,z)
    if (rescaling):
        img_data = img_data / 3 * 1e3
    img_data = img_data[np.newaxis, ...]   # (1,x,y,z)

    return img_data, ras_idx


def load_3D_images(root_dir, block_size=64, pad_size=1, train_ratio=0.80, with_norm=True):
    all_t1w = np.zeros([1, 1, block_size, block_size, block_size])
    all_fa = np.zeros([1, 1, block_size, block_size, block_size])
    all_md = np.zeros([1, 1, block_size, block_size, block_size])

    # loop all dirs under root dir
    for subject_dir in sorted(d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))):
        t1w_img, tmp  = load_image(root_dir, subject_dir, cf.T1W_FILE_NAME)
        fa_img, tmp   = load_image(root_dir, subject_dir, cf.FA_FILE_NAME)
        md_img, tmp   = load_image(root_dir, subject_dir, cf.MD_FILE_NAME, rescaling=True) 
        mask_img, tmp = load_image(root_dir, subject_dir, cf.MASK_FILE_NAME)

        # normlize t1w
        if (with_norm):
            t1w_img = zscore_normalize(t1w_img, mask_img)

        t1w_blocks, _ = extract_blocks(t1w_img, mask_img, size=block_size)
        fa_blocks, _  = extract_blocks(fa_img, mask_img, size=block_size)
        md_blocks, _  = extract_blocks(md_img, mask_img, size=block_size)

        all_t1w = np.concatenate([all_t1w, t1w_blocks], axis = 0)
        all_fa  = np.concatenate([all_fa,  fa_blocks],  axis = 0)
        all_md  = np.concatenate([all_md,  md_blocks],  axis = 0)

    all_t1w = all_t1w[1:,:,:,:,:] 
    all_fa  = all_fa[1:,:,:,:,:]
    all_md  = all_md[1:,:,:,:,:]

    data_sz = np.shape(all_t1w)
    n = data_sz[0]
    train_sample = int(n*train_ratio)
    input_train = all_t1w[:train_sample,:,:,:,:]
    input_val   = all_t1w[train_sample:,:,:,:,:]

    output = np.concatenate([all_fa, all_md], axis = 1)
    output_train = output[:train_sample,:,:,:,:]
    output_val   = output[train_sample:,:,:,:,:]

    if (train_ratio == 1.00) : 
        input_val = None; output_val = None

    return input_train, input_val, output_train, output_val


def save_image(img, img_path):
    print(img_path)
    affine = np.eye(4)

    # Create a NIfTI image
    nifti_img = nib.Nifti1Image(img, affine)

    # Save the image to disk
    nib.save(nifti_img, img_path)


def compute_ssim_mean(pred, true):
    ssim_vals = []
    for i in range(true.shape[2]):  # Z-dim
        dr = true[:, :, i].max() - true[:, :, i].min()

        if dr == 0:
            dr = 1.0

        ssim_val = ssim(pred[:, :, i], true[:, :, i], data_range=dr)
        ssim_vals.append(ssim_val)

    return np.mean(ssim_vals)


def make_loaders(
    input_train: np.ndarray, output_train: np.ndarray,
    input_val: np.ndarray = None, output_val: np.ndarray = None,
    batch_size: int = 10, num_workers: int = 0, augment: bool = True
):
    train_ds = Dataset(input_train, output_train, augment=augment)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = None
    if input_val is not None and output_val is not None:
        val_ds = Dataset(input_val, output_val, augment=False)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

    return train_loader, val_loader


def training(model, train_loader, val_loader, best_model_path, optimizer, criterion, device, epochs=30):
    best_val_loss = float("inf")
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(epochs):
        # ---- TRAIN ------
        start_time = time.time()
        model.train()
        print('Epoch:', epoch+1)
        running_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * xb.size(0)
        train_loss = running_train_loss / len(train_loader.dataset)

        # ---- VALIDATION ------
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                running_val_loss += loss.item() * xb.size(0)
        val_loss = running_val_loss / len(val_loader.dataset)

        print(f" Epoch {epoch+1:03d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, Time = {time.time()-start_time}")

        # ---- CHECKPOINT / EARLY STOP ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  Improved val loss. Model saved (best_val_loss={best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")

    print(f"Training complete. Best validation loss = {best_val_loss:.6f}, Best model = {best_model_path}, Time = {time.time()-start_time}")

def train(model, best_model, root_dir, with_norm=True, l1_loss=True, block_size=64, batch_size=10):
    input_train, input_val, output_train, output_val = load_3D_images(root_dir, with_norm=with_norm, block_size=block_size)
    train_loader, val_loader = make_loaders(input_train, output_train, input_val, output_val, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=cf.LR)
    if (l1_loss):
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    training(model, train_loader, val_loader, cf.BEST_MODEL, optimizer, criterion, cf.DEVICE)


def test(model, best_model, root_dir, with_norm=True, block_size=64):
    state = torch.load(best_model, map_location="cpu")
    model.load_state_dict(state, strict=False)
    criterion = nn.L1Loss()

    fa_all_mse, fa_all_mae, fa_all_rmse, fa_all_r, fa_all_ssim = [], [], [], [], []
    md_all_mse, md_all_mae, md_all_rmse, md_all_r, md_all_ssim = [], [], [], [], []

    # loop all dirs under root dir
    for subject_dir in sorted(d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))):

        t1w_img, _  = load_image(root_dir, subject_dir, cf.T1W_FILE_NAME)
        fa_img, _   = load_image(root_dir, subject_dir, cf.FA_FILE_NAME)
        md_img, _   = load_image(root_dir, subject_dir, cf.MD_FILE_NAME, rescaling=True) 
        mask_img, _ = load_image(root_dir, subject_dir, cf.MASK_FILE_NAME)

        # normlize t1w
        if (with_norm):
            t1w_img = zscore_normalize(t1w_img, mask_img)

        t1w_blocks, ind_block = extract_blocks(t1w_img, mask_img, size=block_size)
        fa_blocks, _  = extract_blocks(fa_img, mask_img, size=block_size)
        md_blocks, _  = extract_blocks(md_img, mask_img, size=block_size)

        net_in = torch.from_numpy(t1w_blocks)
        output_gt = np.concatenate([fa_blocks,  md_blocks], axis = 1)
        y_gt = torch.from_numpy(output_gt)

        with torch.no_grad():
            y_pred = model(net_in.to(device=cf.DEVICE,dtype=torch.float))
            y_gt = y_gt.to(device=cf.DEVICE,dtype=torch.float)
            test_loss = criterion(y_pred, y_gt)

            y_pred = y_pred.cpu().numpy()

        y_pred = assemble_blocks(y_pred, ind_block, mask_img)
        fa_pred_out = y_pred[0,:,:,:]
        md_pred_out = y_pred[1,:,:,:]
        fa_gt_out = fa_img[0,:,:,:]
        md_gt_out = md_img[0,:,:,:]

        mse = np.mean((fa_pred_out - fa_gt_out)**2)
        mae = np.mean(np.abs(md_pred_out - md_gt_out))
        rmse = np.sqrt(mse)
        r = np.corrcoef(fa_pred_out.flatten(), fa_gt_out.flatten())[0,1]
        ssim_mean = compute_ssim_mean(fa_pred_out, fa_gt_out)

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
        ssim_mean = compute_ssim_mean(md_pred_out, md_gt_out)

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


def set_fixed_seed(seed: int = 42):
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False