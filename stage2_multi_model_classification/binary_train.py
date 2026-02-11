import torch
import time
import numpy as np
import utils as ut
import config as cf
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from simplecnn import SimpleCNN
from sklearn.metrics import f1_score

# --------------------------
# Binary Classifier Training
# --------------------------

device = cf.DEVICE
train_loader, val_loader = ut.load_train_data(num_class=2)
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
start_time = 0
train_start_time = time.time()

for epoch in range(cf.ST2_BINARY_EPOCHS):
    # --- TRAIN ---
    start_time = time.time()
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

        print(f" Epoch {epoch+1:03d} - Train: Loss = {train_loss:.6f}, Correct = {train_acc:.6f};  Val : Loss = {val_loss:.6f}, Correct = {val_acc:.6f}； val_macro_f1 = {val_f1}; Time = {time.time()-start_time}")

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

time = time.time()-train_start_time
print(f"Training complete. Best validation loss: {best_val_loss:.6f},  time : {time/60}m {time%60}s")
