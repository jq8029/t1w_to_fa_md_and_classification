import os
import torch
import numpy as np
import config as cf
import nibabel as nib
import torch.nn.functional as F
from dataset import Dataset
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

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
    std = np.sqrt(((x - mean) ** 2 * mask).sum(axis=(1, 2, 3), keepdims=True) / num)
    std = np.clip(std, a_min=1e-6, a_max=None)

    x_norm = (x - mean) / std
#    x_norm = x_norm * mask  # keep background zero

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


def compute_ssim_mean(pred, true):
    ssim_vals = []
    for i in range(true.shape[2]):  # Z-dim
        dr = true[:, :, i].max() - true[:, :, i].min()

        if dr == 0:
            dr = 1.0

        ssim_val = ssim(pred[:, :, i], true[:, :, i], data_range=dr)
        ssim_vals.append(ssim_val)

    return np.mean(ssim_vals)



def box_from_mask(mask):
    coords = np.where(mask)
    zmin, zmax = coords[1].min(), coords[1].max()
    ymin, ymax = coords[2].min(), coords[2].max()
    xmin, xmax = coords[3].min(), coords[3].max()
    return zmin,zmax,ymin,ymax,xmin,xmax

def compute_windows(src_dim, tgt_dim, shift):
    # src indices: [src_s:src_e) in [0, src_dim)
    # tgt indices: [tgt_s:tgt_e) in [0, tgt_dim)
    # shift = tgt_origin - src_origin
    tgt_s = max(0, shift)
    src_s = max(0, -shift)
    length = min(src_dim - src_s, tgt_dim - tgt_s)
    tgt_e = tgt_s + length
    src_e = src_s + length
    return src_s, src_e, tgt_s, tgt_e

def move_to_center(vol, mask, target_shape):
    # -- ensure mask boolean
    mask = (mask > 0)
    C, Z, Y, X = vol.shape
    _, tz, ty, tx = target_shape

    # bounding box
    zmin, zmax, ymin, ymax, xmin, xmax = box_from_mask(mask)
    # center of bbox (voxel indices)
    x_center = (xmin + xmax) // 2
    y_center = (ymin + ymax) // 2
    z_center = (zmin + zmax) // 2

    # compute center of target array
    tz_c = (tz - 1) // 2
    ty_c = (ty - 1) // 2
    tx_c = (tx - 1) // 2

    shift_z = int(tz_c - z_center)
    shift_y = int(ty_c - y_center)
    shift_x = int(tx_c - x_center)

    # allocate target arrays
    vol_centered = np.zeros((C, tz, ty, tx), dtype=vol.dtype)
    mask_centered = np.zeros((C, tz, ty, tx), dtype=bool)

    sz_s, sz_e, tz_s, tz_e = compute_windows(Z, tz, shift_z)
    sy_s, sy_e, ty_s, ty_e = compute_windows(Y, ty, shift_y)
    sx_s, sx_e, tx_s, tx_e = compute_windows(X, tx, shift_x)

    # place
    vol_centered[:, tz_s:tz_e, ty_s:ty_e, tx_s:tx_e] = vol[:, sz_s:sz_e, sy_s:sy_e, sx_s:sx_e]
    mask_centered[:, tz_s:tz_e, ty_s:ty_e, tx_s:tx_e] = mask[:, sz_s:sz_e, sy_s:sy_e, sx_s:sx_e]

    return vol_centered


def normalize_imge_mean(image, mask):
    vals = image[mask]
    image = (image - vals.mean()) / (vals.std() + 1e-8)
    image[~mask] = 0  
    return image

def bbox_from_mask(mask):
    coords = np.where(mask)
    zmin, zmax = coords[0].min(), coords[0].max()
    ymin, ymax = coords[1].min(), coords[1].max()
    xmin, xmax = coords[2].min(), coords[2].max()
    return zmin,zmax,ymin,ymax,xmin,xmax
    

def center_volume_to_target(vol, mask):
  
    # -- ensure mask boolean
    mask = (mask > 0)
    Z, Y, X = vol.shape
    tz, ty, tx = vol.shape
    target = (tz, ty, tx) 

    # bounding box
    zmin, zmax, ymin, ymax, xmin, xmax = bbox_from_mask(mask)
    # center of bbox (voxel indices)
    x_center = (xmin + xmax) // 2
    y_center = (ymin + ymax) // 2
    z_center = (zmin + zmax) // 2

    # compute center of target array
    tz_c = (target[0] - 1) // 2
    ty_c = (target[1] - 1) // 2
    tx_c = (target[2] - 1) // 2

    # compute shift required: new_index = old_index + shift  => want old_center + shift = target_center
    shift_z = int(tz_c - z_center)
    shift_y = int(ty_c - y_center)
    shift_x = int(tx_c - x_center)

    # allocate target arrays
    vol_centered = np.zeros(target, dtype=vol.dtype)
    mask_centered = np.zeros(target, dtype=bool)

    # compute source and dest windows (clamp to bounds)
    def compute_windows(src_dim, tgt_dim, shift):
        # src indices: [src_s:src_e) in [0, src_dim)
        # tgt indices: [tgt_s:tgt_e) in [0, tgt_dim)
        # shift = tgt_origin - src_origin
        tgt_s = max(0, shift)
        src_s = max(0, -shift)
        length = min(src_dim - src_s, tgt_dim - tgt_s)
        tgt_e = tgt_s + length
        src_e = src_s + length
        return src_s, src_e, tgt_s, tgt_e

    sz_s, sz_e, tz_s, tz_e = compute_windows(Z, target[0], shift_z)
    sy_s, sy_e, ty_s, ty_e = compute_windows(Y, target[1], shift_y)
    sx_s, sx_e, tx_s, tx_e = compute_windows(X, target[2], shift_x)

    # place
    vol_centered[tz_s:tz_e, ty_s:ty_e, tx_s:tx_e] = vol[sz_s:sz_e, sy_s:sy_e, sx_s:sx_e]
    mask_centered[tz_s:tz_e, ty_s:ty_e, tx_s:tx_e] = mask[sz_s:sz_e, sy_s:sy_e, sx_s:sx_e]

    shift_vox = (shift_z, shift_y, shift_x)
    return vol_centered


def load_2D_images(source_path, label_value, shuffle=True, train_ratio=0.8, num_slices=30, interval=2):

    count = 0
    skip_list = []
    # use list to add slices
    x_list = []

    for subject_dir in sorted(d for d in os.listdir(source_path)
                if os.path.isdir(os.path.join(source_path, d))):

        count += 1

        # --- Load NIfTI ---
        t1w_img, ras_idx  = load_image(source_path, subject_dir, cf.T1W_FILE_NAME)  # (N,X,Y,Z)
        fa_img, tmp       = load_image(source_path, subject_dir, cf.FA_FILE_NAME)
        md_img, tmp       = load_image(source_path, subject_dir, cf.MD_FILE_NAME, rescaling=True) 
        mask_img, tmp     = load_image(source_path, subject_dir, cf.MASK_FILE_NAME)

        # only normlize t1w
        t1w_img = zscore_normalize(t1w_img, mask_img)

        # --- Flip if needed ---
        if (ras_idx == 1):  # I, P, L
            t1w_img   = np.transpose(t1w_img[ :, ::-1, ::-1, ::-1], (0, 3, 2, 1))
            fa_img    = np.transpose(fa_img[ :, ::-1, ::-1, ::-1], (0, 3, 2, 1))
            md_img    = np.transpose(md_img[ :, ::-1, ::-1, ::-1], (0, 3, 2, 1))
            mask_img  = np.transpose(mask_img[ :, ::-1, ::-1, ::-1], (0, 3, 2, 1))
        if (ras_idx == 2):  # P, I, R
            t1w_img   = np.transpose(t1w_img[ :, ::-1, ::-1, ::-1], (0, 3, 1, 2))
            fa_img    = np.transpose(fa_img[ :, ::-1, ::-1, ::-1], (0, 3, 1, 2))
            md_img    = np.transpose(md_img[ :, ::-1, ::-1, ::-1], (0, 3, 1, 2))
            mask_img  = np.transpose(mask_img[ :, ::-1, ::-1, ::-1], (0, 3, 1, 2))
       
        # shape after adjustments
        _, tH, tW, tD = t1w_img.shape
        _, fH, fW, fD = fa_img.shape
        _, mH, mW, mD = md_img.shape

        coords = np.where(mask_img)
        xmin, xmax = coords[3].min(), coords[3].max()

        t1w_img = move_to_center(t1w_img, mask_img, t1w_img.shape)
        fa_img = move_to_center(fa_img, mask_img, fa_img.shape)
        md_img = move_to_center(md_img, mask_img, md_img.shape)

        # --- Pre-allocate padded cube once ---
        vol = np.zeros((3, 1, 180, 280, 180), dtype=np.float32)
        vol[0, 0, :tH, :tW, :tD] = t1w_img
        vol[1, 0, :fH, :fW, :fD] = fa_img
        vol[2, 0, :mH, :mW, :mD] = md_img

        xc = (xmin + xmax) // 2

        center_slice = xc
        slice_idx = list(range(center_slice-num_slices,center_slice+num_slices,interval))
        total_slices_per_subj = len(slice_idx)

        # --- Extract slices ---
        for s in slice_idx:
            slice = vol[:,:,s,:,:]
            x_list.append(slice)    # (3,1,280,180)

        if count % 20 == 0:
            print(f"  Loaded {count} subjects. x_list = {len(x_list)}")

    x_list = np.stack(x_list, axis=0)       # (N, C, 1, H, W)
    x_list = np.squeeze(x_list, axis=2)     # (N, C, H, W)

    # split first
    N = len(x_list)
    train_N = int(N * train_ratio)
    input_train = x_list[:train_N]
    input_val = x_list[train_N:]

    if (shuffle):
        idx_train = np.random.permutation(len(input_train))
        idx_val = np.random.permutation(len(input_val))

        input_train = input_train[idx_train]
        input_val = input_val[idx_val]

    # labels
    output = torch.full((N,), float(label_value))
    output_train = output[:train_N]
    output_val = output[train_N:]

    print(f'Loaded {count} subjects in {source_path} : {input_train.shape}, {input_val.shape}, {output_train.shape}, {output_val.shape} ')

    return input_train, input_val, output_train, output_val



def make_train_loaders(
    X_train: np.ndarray, Y_train: np.ndarray,
    X_val: np.ndarray = None, Y_val: np.ndarray = None,
    batch_size: int = 1, num_workers: int = 0, augment: bool = True
):
    train_ds = Dataset(X_train, Y_train, augment=augment)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = None
    if X_val is not None and Y_val is not None:
        val_ds = Dataset(X_val, Y_val, augment=False)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    return train_loader, val_loader

def make_test_loaders(
    X_test: np.ndarray, Y_test: np.ndarray,
    batch_size: int = 1, num_workers: int = 0, augment: bool = False
):
    test_ds = Dataset(X_test, Y_test, augment=augment)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return test_loader

def load_train_data(num_class=2):
    # Step 1 : load all nc data
    # load training data
    label = 0
    input_train, tmp1, output_train, tmp2 = load_2D_images(cf.ST2_TRAIN_NC_DIR, label, True, 1)
    all_input_train = input_train
    all_output_train = output_train
    input_val, tmp1, output_val, tmp2 = load_2D_images(cf.ST2_VAL_NC_DIR, label, False, 1)
    all_input_val = input_val
    all_output_val = output_val
    label = label + 1

    if (num_class > 2):
        input_train, tmp1, output_train, tmp2 = load_2D_images(cf.ST2_TRAIN_MCI_DIR, label, True, 1)
        all_input_train = np.concatenate([all_input_train, input_train], axis = 0)
        all_output_train = np.concatenate([all_output_train, output_train], axis = 0)
        input_val, tmp1, output_val, tmp2 = load_2D_images(cf.ST2_VAL_MCI_DIR, label, False, 1)
        all_input_val = np.concatenate([all_input_val, input_val], axis = 0)
        all_output_val = np.concatenate([all_output_val, output_val], axis = 0)   
        label = label + 1     

    # Step 2 : load ad data
    input_train, tmp1, output_train, tmp2 = load_2D_images(cf.ST2_TRAIN_AD_DIR, label, True, 1)
    all_input_train = np.concatenate([all_input_train, input_train], axis = 0)
    all_output_train = np.concatenate([all_output_train, output_train], axis = 0)
    input_val, tmp1, output_val, tmp2 = load_2D_images(cf.ST2_VAL_AD_DIR, label, False, 1)
    all_input_val = np.concatenate([all_input_val, input_val], axis = 0)
    all_output_val = np.concatenate([all_output_val, output_val], axis = 0)

    return make_train_loaders(all_input_train, all_output_train, all_input_val, all_output_val, batch_size = cf.ST2_BINARY_BATCH_SIZE, num_workers=0)

def load_test_data(num_class=2):
    # Step 1 : load all nc data
    # load training data
    label = 0
    input_train, input_val, output_train, output_val = load_2D_images(cf.ST2_TRAIN_NC_DIR, label, True, 1)
    test_input_train = input_train
    test_input_val = input_val
    test_output_train = output_train
    test_output_val = output_val
    label = label + 1

    if (num_class > 2):
        input_train, input_val, output_train, output_val = load_2D_images(cf.ST2_TRAIN_MCI_DIR, label, True, 1)
        test_input_train = np.concatenate([test_input_train, input_train], axis = 0)
        test_input_val = np.concatenate([test_input_val, input_val], axis = 0)
        test_output_train = np.concatenate([test_output_train, output_train], axis = 0)
        test_output_val = np.concatenate([test_output_val, output_val], axis = 0)  
        label = label + 1     

    # Step 2 : load ad data
    input_train, input_val, output_train, output_val = load_2D_images(cf.ST2_TRAIN_AD_DIR, label, True, 1)
    test_input_train = np.concatenate([test_input_train, input_train], axis = 0)
    test_input_val = np.concatenate([test_input_val, input_val], axis = 0)
    test_output_train = np.concatenate([test_output_train, output_train], axis = 0)
    test_output_val = np.concatenate([test_output_val, output_val], axis = 0)

    return make_test_loaders(test_input_train, test_output_train, batch_size = cf.ST2_BINARY_BATCH_SIZE, num_workers=0)