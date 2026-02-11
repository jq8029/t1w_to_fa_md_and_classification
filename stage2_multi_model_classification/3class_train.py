import torch
import config as cf
import torch.nn as nn
import utils as ut
import torch.optim as optim
from simplecnn import SimpleCNN
from sklearn.metrics import f1_score


# -------------------------------
# Three-Class Classifier Training
# -------------------------------

train_loader, val_loader = ut.load_train_data(num_class=3)

# --- Model / optimizer / loss ---
device = cf.DEVICE
model = SimpleCNN(3, 3).to(device)
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