import os
import torch
import numpy as np
import config as cf
import nibabel as nib
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


def load_3D_images(root_dir, block_size=64, pad_size=1, train_ratio=0.80):
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
        t1w_img = zscore_normalize(t1w_img, mask_img)

        t1w_blocks, _ = extract_blocks(t1w_img, mask_img)
        fa_blocks, _  = extract_blocks(fa_img, mask_img)
        md_blocks, _  = extract_blocks(md_img, mask_img)

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
    batch_size: int = 1, num_workers: int = 0, augment: bool = True
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