import os
import torch
import numpy as np
import config as cf
import nibabel as nib
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset import Dataset
from simplecnn import SimpleCNN
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import f1_score
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report
)

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


def load_2D_images(source_path, label_value, shuffle=True, train_ratio=0.8, num_slices=30, interval=2, single=False):

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
        if (single):
            vol = np.zeros((1, 1, 180, 280, 180), dtype=np.float32)
            vol[0, 0, :tH, :tW, :tD] = t1w_img
        else:
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

def load_train_data(num_class=2, single=False):
    # Step 1 : load all nc data
    # load training data
    label = 0
    input_train, tmp1, output_train, tmp2 = load_2D_images(cf.ST2_TRAIN_NC_DIR, label, True, 1, single=single)
    all_input_train = input_train
    all_output_train = output_train
    input_val, tmp1, output_val, tmp2 = load_2D_images(cf.ST2_VAL_NC_DIR, label, False, 1, single=single)
    all_input_val = input_val
    all_output_val = output_val
    label = label + 1

    if (num_class > 2):
        input_train, tmp1, output_train, tmp2 = load_2D_images(cf.ST2_TRAIN_MCI_DIR, label, True, 1, single=single)
        all_input_train = np.concatenate([all_input_train, input_train], axis = 0)
        all_output_train = np.concatenate([all_output_train, output_train], axis = 0)
        input_val, tmp1, output_val, tmp2 = load_2D_images(cf.ST2_VAL_MCI_DIR, label, False, 1, single=single)
        all_input_val = np.concatenate([all_input_val, input_val], axis = 0)
        all_output_val = np.concatenate([all_output_val, output_val], axis = 0)   
        label = label + 1     

    # Step 2 : load ad data
    input_train, tmp1, output_train, tmp2 = load_2D_images(cf.ST2_TRAIN_AD_DIR, label, True, 1, single=single)
    all_input_train = np.concatenate([all_input_train, input_train], axis = 0)
    all_output_train = np.concatenate([all_output_train, output_train], axis = 0)
    input_val, tmp1, output_val, tmp2 = load_2D_images(cf.ST2_VAL_AD_DIR, label, False, 1, single=single)
    all_input_val = np.concatenate([all_input_val, input_val], axis = 0)
    all_output_val = np.concatenate([all_output_val, output_val], axis = 0)

    return make_train_loaders(all_input_train, all_output_train, all_input_val, all_output_val, batch_size = cf.ST2_BINARY_BATCH_SIZE, num_workers=0)

def load_test_data(num_class=2, single=False):
    # Step 1 : load all nc data
    # load training data
    label = 0
    input_train, input_val, output_train, output_val = load_2D_images(cf.ST2_TRAIN_NC_DIR, label, True, 1, single=single)
    test_input_train = input_train
    test_input_val = input_val
    test_output_train = output_train
    test_output_val = output_val
    label = label + 1

    if (num_class > 2):
        input_train, input_val, output_train, output_val = load_2D_images(cf.ST2_TRAIN_MCI_DIR, label, True, 1, single=single)
        test_input_train = np.concatenate([test_input_train, input_train], axis = 0)
        test_input_val = np.concatenate([test_input_val, input_val], axis = 0)
        test_output_train = np.concatenate([test_output_train, output_train], axis = 0)
        test_output_val = np.concatenate([test_output_val, output_val], axis = 0)  
        label = label + 1     

    # Step 2 : load ad data
    input_train, input_val, output_train, output_val = load_2D_images(cf.ST2_TRAIN_AD_DIR, label, True, 1, single=single)
    test_input_train = np.concatenate([test_input_train, input_train], axis = 0)
    test_input_val = np.concatenate([test_input_val, input_val], axis = 0)
    test_output_train = np.concatenate([test_output_train, output_train], axis = 0)
    test_output_val = np.concatenate([test_output_val, output_val], axis = 0)

    return make_test_loaders(test_input_train, test_output_train, batch_size = cf.ST2_BINARY_BATCH_SIZE, num_workers=0)


def train_3class(single=False):
    train_loader, val_loader = load_train_data(num_class=3, single=single)

    # --- Model / optimizer / loss ---
    device = cf.DEVICE
    model = SimpleCNN(3, 3).to(device)
    if (single):
        model = SimpleCNN(1, 3).to(cf.DEVICE)
    else:
        model = SimpleCNN(3, 3).to(cf.DEVICE)
    opt = optim.Adam(model.parameters(), lr=cf.ST2_3CLASS_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='min',          
        factor=0.5,           
        patience=3,           
        min_lr=1e-6           
    )
    weights = torch.tensor([1.1, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.02)

    # --- Training config ---
    best_val_loss = float("inf")
    best_val_f1 = 0
    epochs_no_improve = 0
    NC, MCI, AD = 0, 1, 2

    for epoch in range(cf.ST2_3CLASS_EPOCHS):
        # ---------- TRAIN ----------
        model.train()
        opt.zero_grad()
        print('Epoch: ', epoch+1)

        running_train_loss = 0.0
        running_train_correct = 0
        count = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.long().to(device)
            opt.zero_grad()
                
            pred = model(imgs)
            loss = criterion(pred, labels)
            loss.backward()
            opt.step()

            running_train_loss += loss.item() * imgs.size(0)
            running_train_correct += (pred.argmax(dim=1) == labels).sum().item()

        train_loss = running_train_loss / len(train_loader.dataset)
        train_acc = running_train_correct / len(train_loader.dataset)

        # --- VALIDATION ---
        model.eval()
        running_val_loss = 0.0
        running_val_correct = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.long().to(device)

                logits  = model(imgs)
                loss = criterion(pred, labels)
                probs = torch.softmax(logits , dim=1)  

                preds = torch.zeros(len(probs), dtype=torch.long)
                for i, p in enumerate(probs):
                    if p[2] >= cf.ST2_3CLASS_AD_POSSIBLE_THRESHOLD:
                        preds[i] = AD
                    elif p[0] >= cf.ST2_3CLASS_MCI_POSSIBLE_THRESHOLD:
                        preds[i] = MCI
                    else:
                        preds[i] = NC

                running_val_loss += loss.item() * imgs.size(0)
                running_val_correct += (pred.argmax(dim=1) == labels).sum().item()

                all_preds.append(pred.argmax(dim=1).cpu())
                all_labels.append(labels.cpu())

        val_loss = running_val_loss / len(val_loader.dataset)
        val_acc = running_val_correct / len(val_loader.dataset)
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        scheduler.step(val_loss) 

        print(f" Epoch {epoch+1:03d} - Train: Loss = {train_loss:.6f}, Correct = {train_acc:.6f};  Val : Loss = {val_loss:.6f}, Correct = {val_acc:.6f}; f1={val_f1}")

        # best model based on f1 score
        if val_f1 > best_val_f1 :
            torch.save(model.state_dict(), cf.ST2_3CLASS_BEST_MODEL)
            print(f"    Saved best model : {cf.ST2_3CLASS_BEST_MODEL}, f1={val_f1}")
            best_val_f1 = val_f1

        # early stop based on loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            print(f"    Improved val loss. Model saved (best_val_loss={best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"    No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= cf.ST2_3CLASS_PATIENCE:
            print("Early stopping triggered!!")
            break

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")


def test_3class(single=False):
    test_loader = load_test_data(num_class=3, single=single)
    if (single):
        model = SimpleCNN(1, 3).to(cf.DEVICE)
    else:
        model = SimpleCNN(3, 3).to(cf.DEVICE)
    state = torch.load(cf.ST2_3CLASS_BEST_MODEL, map_location="cpu")
    model.load_state_dict(state, strict=False)
    criterion = nn.CrossEntropyLoss()

    test_loss_total = 0 
    correct_total = 0
    y_true = []
    y_pred = []
    NC, MCI, AD = 0, 1, 2

    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(cf.DEVICE)
            labels = labels.long().to(cf.DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            test_loss_total += loss.item()

            prob = torch.softmax(outputs, dim=1)  # [n, 3]

            # slice decision
            slice_preds = []
            for i in range(prob.shape[0]):
                p_nc, p_mci, p_ad = prob[i]

                if p_ad >= cf.ST2_3CLASS_AD_POSSIBLE_THRESHOLD:
                    slice_preds.append(AD)
                elif p_mci >= cf.ST2_3CLASS_MCI_POSSIBLE_THRESHOLD:
                    slice_preds.append(MCI)
                else:
                    slice_preds.append(NC)

            slice_preds = torch.tensor(slice_preds)

            # count the number of ad/mci slices in the same subject
            ad_count  = (slice_preds == AD).sum().item()
            mci_count = (slice_preds == MCI).sum().item()

            # subject decision
            if ad_count >= cf.ST2_3CLASS_AD_SUBJECT_THRESHOLD:
                subject_pred = AD
            elif mci_count >= cf.ST2_3CLASS_MCI_SUBJECT_THRESHOLD:
                subject_pred = MCI
            else:
                subject_pred = NC

            subject_label = labels[0].item()

            y_true.append(subject_label)
            y_pred.append(subject_pred)

            correct_total += int(subject_pred == subject_label)

    # ---- Subject-level metrics ----
    test_loss = test_loss_total / len(test_loader)
    test_acc = correct_total / len(y_true)

    print("===== Subject-level Evaluation =====")
    print(f"Test Loss      : {test_loss:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print("------------------------------------")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("------------------------------------")
    print(classification_report(
        y_true, y_pred,
        target_names=["NC", "MCI", "AD"],
        digits=3
    ))
    print("------------------------------------")


def train_binary(single=False):
    device = cf.DEVICE
    train_loader, val_loader = load_train_data(num_class=2, single=single)
    if (single):
        model = SimpleCNN(1, 1).to(device)
    else:
        model = SimpleCNN(3, 1).to(device)
    opt = optim.Adam(model.parameters(), lr=cf.ST2_BINARY_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='min',          
        factor=0.5,           
        patience=3,           
        min_lr=1e-6           
    )

    pos_weight = torch.tensor(0.45).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Training config ---
    best_val_loss = float("inf")
    best_val_f1 = 0
    epochs_no_improve = 0

    for epoch in range(cf.ST2_BINARY_EPOCHS):
        # --- TRAIN ---
        model.train()
        opt.zero_grad()
        print('Epoch: ', epoch+1)

        running_train_loss = 0.0
        running_train_correct = 0
        count = 0

        for imgs, labels in train_loader:
            labels = labels.float().unsqueeze(1).to(device)
            imgs = imgs.to(device)
            opt.zero_grad()
                
            pred = model(imgs)
            loss = criterion(pred, labels)
            loss.backward()
            opt.step()

            running_train_loss += loss.item() * imgs.size(0)
            prob = torch.sigmoid(pred)
            out = (prob > cf.ST2_BINARY_POSSIBLE_THRESHOLD).float()

            running_train_correct += (out == labels).sum().item()

        train_loss = running_train_loss / len(train_loader.dataset)
        train_acc = running_train_correct / len(train_loader.dataset)

        # --- VALIDATION ---
        model.eval()
        all_preds = []
        all_labels = []
        subject_preds = []
        subject_labels = []
        running_val_loss = 0.0
        running_val_correct = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                labels = labels.float().unsqueeze(1).to(device)
                imgs = imgs.to(device)

                pred = model(imgs)                
                loss = criterion(pred, labels)
                running_val_loss += loss.item() * imgs.size(0)

                probs = torch.sigmoid(pred)
                slice_preds = (probs > cf.ST2_BINARY_POSSIBLE_THRESHOLD).int()

                # --- subject-level voting ---
                num_positive = slice_preds.sum().item()
                subject_pred = int(num_positive >= cf.ST2_BINARY_SUBJECT_THRESHOLD)

                subject_label = int(labels[0].item())

                subject_preds.append(subject_pred)
                subject_labels.append(subject_label)

            val_loss = running_val_loss / len(val_loader.dataset)
            val_f1 = f1_score(subject_labels, subject_preds, average='macro')
            val_acc = np.mean(np.array(subject_preds) == np.array(subject_labels))

            scheduler.step(val_loss) 

            print(f" Epoch {epoch+1:03d} - Train: Loss = {train_loss:.6f}, Correct = {train_acc:.6f};  Val : Loss = {val_loss:.6f}, Correct = {val_acc:.6f}； val_macro_f1 = {val_f1}")

            # best model based on f1 score
            if val_f1 > best_val_f1 and val_loss < 0.6:
                torch.save(model.state_dict(), cf.ST2_BINARY_BEST_MODEL)
                best_val_f1 = val_f1
                print(f"    Saved best model : {cf.ST2_BINARY_BEST_MODEL}, f1={val_f1}")

            # early stop based on loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                print(f"    Improved val loss. Model saved (best_val_loss={best_val_loss:.6f})")
            else:
                epochs_no_improve += 1
                print(f"    No improvement for {epochs_no_improve} epoch(s)")

            if epochs_no_improve >= cf.ST2_BINARY_PATIENCE:
                print("Early stopping triggered!!")
                break

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")


def test_binary(single=False):
    test_loader = load_test_data(num_class=2, single=single)
    device = cf.DEVICE
    if (single):
        model = SimpleCNN(1, 1).to(device)
    else:
        model = SimpleCNN(3, 1).to(device)
    model.load_state_dict ( torch.load(cf.ST2_BINARY_BEST_MODEL) )
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    NC, AD = 0, 1

    y_true = [] 
    y_pred = []
    y_score = []
    test_loss_total = 0
    correct_total = 0
    pos_threshold = cf.ST2_BINARY_POSSIBLE_THRESHOLD
    subject_threshold = cf.ST2_BINARY_SUBJECT_THRESHOLD

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device) 

            logits = model(imgs)                           
            loss = criterion(logits, labels)
            test_loss_total += loss.item()

            probs = torch.sigmoid(logits).view(-1)       
            num_positive = (probs >= pos_threshold).sum().item()

            # Subject-level voting
            if num_positive >= subject_threshold:
                final_pred = AD
            else:
                final_pred = NC

            final_label = int(labels[0].item())       

            y_true.append(final_label)
            y_pred.append(final_pred)
            y_score.append(probs.mean().item())       

            correct_total += int(final_pred == final_label)

    # ===== Metrics =====
    test_loss = test_loss_total / len(test_loader)
    test_acc = correct_total / len(y_true)

    print("=== Subject-level Evaluation ===")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=3))

    auc = roc_auc_score(y_true, y_score)
    print(f"Subject-Level AUC: {auc:.4f}")