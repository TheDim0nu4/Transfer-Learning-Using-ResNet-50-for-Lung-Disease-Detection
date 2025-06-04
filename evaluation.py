import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve




def evaulation( resnet50, test_loader, optimal_thresholds, device ):

    resnet50.eval()            

    all_test_labels, all_test_preds = [], []

    for batch in test_loader:

        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():  
            outputs = resnet50(images)

        all_test_labels.append(labels.cpu().numpy())
        all_test_preds.append(torch.sigmoid(outputs).cpu().detach().numpy())


    all_test_labels = np.concatenate(all_test_labels, axis=0)
    all_test_preds = np.concatenate(all_test_preds, axis=0)


    test_auc = roc_auc_score(all_test_labels, all_test_preds, average='macro')

    print(f"Test. AUC-ROC: {test_auc}")



    thresholds = np.array(optimal_thresholds)[None, :]
    binary_preds = (all_test_preds >= thresholds).astype(int)

    sensitivities, specificities = [], []
    fpr_list, tpr_list = [], []


    for idx in range(all_test_labels.shape[1]):

        y_true = all_test_labels[:, idx]
        y_pred = binary_preds[:, idx]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

        print(f"Class {idx}: Sensitivity = {sensitivity}, Specificity = {specificity}")


        fpr, tpr, _ = roc_curve(y_true, all_test_preds[:, idx])
        fpr_list.append(fpr.tolist())
        tpr_list.append(tpr.tolist())


    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)

    print(f"\nMean per-class Sensitivity: {np.nanmean(sensitivities)}")
    print(f"Mean per-class Specificity: {np.nanmean(specificities)}")



    with open("training_result/auc_test.txt", 'w') as f:
        f.write(f"{test_auc}\n")


    with open("training_result/sensitivity_test.txt", 'w') as f:
        f.write(f"{sensitivities.tolist()}\n")


    with open("training_result/specificity_test.txt", 'w') as f:
        f.write(f"{specificities.tolist()}\n")


    with open("training_result/fpr_roc_curve.txt", "w") as f:
        f.write(f"{fpr_list}\n")

    with open("training_result/tpr_roc_curve.txt", "w") as f:
        f.write(f"{tpr_list}\n")




