import torch
import time
import utils as ut
import config as cf
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from transunet import TransUNet
from dataset import Dataset


def training(model, train_loader, val_loader, best_model_path, optimizer, criterion, device, epochs=200, patience=10):
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

        if epochs_no_improve >= patience and epoch > 100:
            print( "  Early stopping triggered!")
            break

    print(f"Training complete. Best validation loss = {best_val_loss:.6f}, Best model = {best_model_path}, Time = {time.time()-start_time}")



# --- Load Data ---
input_train, input_val, output_train, output_val = ut.load_3D_images(cf.ST1_TRAIN_DIR)
train_loader, val_loader = ut.make_loaders(input_train, output_train, input_val, output_val, batch_size = cf.ST1_BATCH_SIZE, num_workers=cf.NUM_WORKERS)

# --- Model / optimizer / loss ---
model = TransUNet(in_channels=cf.IN_CHANNELS, out_channels=cf.OUT_CHANNELS, base=cf.BASE_CHANNELS).to(cf.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=cf.ST1_LR)
criterion = nn.L1Loss()

# --- Training ---
training(model, train_loader, val_loader, cf.ST1_TRAIN_BEST_MODEL, optimizer, criterion, cf.DEVICE, cf.ST1_EPOCHS, cf.ST1_PATIENCE)