import torch
import numpy as np
import utils as ut
import config as cf
import torch.nn as nn
from simplecnn import SimpleCNN
from sklearn.metrics import (
    confusion_matrix, classification_report
)

# ---------------------------
# Three-Class Classifier Test
# ---------------------------

test_loader = ut.load_test_data(num_class=3)
model = SimpleCNN(3, 3).to(cf.device)
model.load_state_dict ( torch.load(cf.ST2_3CLASS_BEST_MODEL) )
criterion = nn.CrossEntropyLoss()

test_loss_total = 0, correct_total = 0
y_true = [], y_pred = []
NC, MCI, AD = 0, 1, 2

model.eval()
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(cf.device)
        labels = labels.long().to(cf.device)

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
