import torch
import numpy as np
from sklearn.metrics import roc_curve




def get_optimal_thresholds(resnet50, val_loader, device):

    resnet50.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():

        for images, labels in val_loader:

            images, labels = images.to(device), labels.to(device)
            outputs = resnet50(images)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())


    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    optimal_thresholds = []


    for class_idx in range(all_labels.shape[1]):

        labels = all_labels[:, class_idx]
        preds = all_preds[:, class_idx]

        fpr, tpr, thresholds = roc_curve(labels, preds)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        optimal_thresholds.append(best_threshold)

        print(f"Class {class_idx}: Optimal threshold (Youden's J) = {best_threshold:.4f}")


    with open("training_result/optimal_thresholds_youden.txt", 'w') as f:
        f.write(f"{optimal_thresholds}\n")


    return optimal_thresholds
