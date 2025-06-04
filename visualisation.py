import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
from torchvision import transforms
from PIL import Image




def roc_curve(mapping):

    with open("training_result/fpr_roc_curve.txt", "r") as f:
        data_str = f.read()

    fpr_roc_curve = eval(data_str)

    with open("training_result/tpr_roc_curve.txt", "r") as f:
        data_str = f.read()

    tpr_roc_curve = eval(data_str)

    num_classes = 15


    plt.figure(figsize=(10, 8))

    for i in range(num_classes):

        fpr = fpr_roc_curve[i]
        tpr = tpr_roc_curve[i]
        auc_score = auc(fpr, tpr)
        label = f"{i} {mapping[i]} (AUC = {auc_score:.3f})"
        plt.plot(fpr, tpr, label=label)


    plt.plot([0, 1], [0, 1], 'k--', label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def show_training():

    with open("training_result/auc_train.txt") as f:
        data_str = f.read()

    auc_train = [float(x) for x in eval(data_str)]

    with open("training_result/auc_validation.txt") as f:
        data_str = f.read()

    auc_validation = [float(x) for x in eval(data_str)]


    with open("training_result/auc_test.txt") as f:
        data_str = f.read()

    auc_test = eval(data_str)


    epochs = list(range(1, len(auc_train) + 1))

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, auc_train, color='green', label='Train AUC')
    plt.plot(epochs, auc_validation, color='blue', label='Validation AUC')

    plt.scatter(epochs, auc_train, color='green')
    plt.scatter(epochs, auc_validation, color='blue')

    plt.scatter([epochs[-1]], [auc_test], color='red', label=f'Test AUC ({auc_test:.3f})', zorder=5)

    plt.title("AUC over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.grid(True)
    plt.legend()
    plt.xticks(epochs) 
    plt.tight_layout()
    plt.show()




def sensitivity_specificity(mapping):

    with open("training_result/sensitivity_test.txt", "r") as f:
        data_str = f.read()

    sensitivity = eval(data_str)

    with open("training_result/specificity_test.txt", "r") as f:
        data_str = f.read()

    specificity = eval(data_str)


    indices = np.arange(len(sensitivity))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(indices - width/2, sensitivity, width, label='Sensitivity', color='skyblue')
    plt.bar(indices + width/2, specificity, width, label='Specificity', color='orange')

    plt.xlabel('Class')
    plt.ylabel('Value')
    plt.title('Sensitivity and Specificity per Class')


    class_labels = [f"{mapping[i]} ({i})" for i in indices]
    plt.xticks(indices, class_labels, rotation=30, ha='right')


    plt.xticks(indices)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()




def class_distribution(mapping, with_augmentation):

    filepath = "training_result/class_distribution_before_augmentation.txt"

    if with_augmentation:
        filepath = filepath.replace("before", "after")


    with open(filepath, "r") as f:
        data_str = f.read()

    distribution = eval(data_str)


    indices = np.arange(len(distribution))
    class_labels = [f"{mapping[i]} ({i})" for i in indices]

    plt.figure(figsize=(14, 6))
    plt.bar(indices, distribution, color='teal')

    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.title("Class Distribution " + ("After" if with_augmentation else "Before") + " Augmentation")
    plt.xticks(indices, class_labels, rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()




def show_augmentation(filepath_image):

    original_image = Image.open(filepath_image).convert("RGB")

    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=10, translate=(0.06, 0.06)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ])


    augmented_images = [transform(original_image) for _ in range(5)]


    plt.figure(figsize=(15, 5))
    plt.subplot(1, 6, 1)
    plt.imshow(original_image)
    plt.title("Original")
    plt.axis("off")

    for i, img in enumerate(augmented_images):
        plt.subplot(1, 6, i + 2)
        plt.imshow(img)
        plt.title(f"Augmented {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()










if __name__ == "__main__":


    mapping = {
        0: "No Finding", 1: "Atelectasis", 2: "Cardiomegaly", 3: "Effusion", 4: "Infiltration",
        5: "Mass", 6: "Nodule", 7: "Pneumonia", 8: "Pneumothorax", 9: "Consolidation",
        10: "Edema", 11: "Emphysema", 12: "Fibrosis", 13: "Pleural_Thickening", 14: "Hernia"
    }


    roc_curve(mapping)
    show_training()
    sensitivity_specificity(mapping)

    class_distribution(mapping, with_augmentation=False)
    class_distribution(mapping, with_augmentation=True)

    show_augmentation("./example_image.jpg")

