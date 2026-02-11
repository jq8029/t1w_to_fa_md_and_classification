import os
import torch
import numpy as np
import utils as ut
import config as cf
import torch.nn as nn
from simplecnn import SimpleCNN
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report
)

# ----------------------
# Binary Classifier Test
# ----------------------

test_loader = ut.load_test_data(num_class=2)
device = cf.DEVICE
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