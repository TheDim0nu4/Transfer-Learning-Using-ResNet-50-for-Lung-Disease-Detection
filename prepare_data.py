from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


from datasets import load_dataset
from ChestXRayDataset import ChestXRayDataset
from torchvision import transforms
from torch.utils.data import  DataLoader
from augmented_dataset import augment_classes
import numpy as np




def get_loaders(batch_size, data_split):

    dataset = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", "image-classification", trust_remote_code=True)



    total_size = len(dataset["train"])
    train_size = int(data_split["train"] * total_size)
    val_size = int(data_split["validation"] * total_size)
    test_size = total_size - train_size - val_size

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    shuffled = dataset['train'].shuffle()
    train_data = shuffled.select(range(train_size))
    val_data = shuffled.select(range(train_size, train_size + val_size))
    test_data = shuffled.select(range(train_size + val_size, total_size))


    num_classes = 15

    class_distribution = np.zeros(num_classes, dtype=np.int32)

    for sample in train_data:
        for label in sample['labels']:
            class_distribution[label] += 1

    with open("training_result/class_distribution_before_augmentation.txt", "w") as f:
        f.write(f"{class_distribution.tolist()}\n")



    augmentation_dict = {14:6, 7:5, 12:5, 10:4, 11:4, 2:4, 13:3, 9:2, 8:2, 5:2, 6:2}
    augmented_train_data = augment_classes(train_data, augmentation_dict)


   
    class_distribution = np.zeros(num_classes, dtype=np.int32)

    for sample in augmented_train_data:
        for label in sample['labels']:
            class_distribution[label] += 1

    with open("training_result/class_distribution_after_augmentation.txt", "w") as f:
        f.write(f"{class_distribution.tolist()}\n")



    train_dataset = ChestXRayDataset(augmented_train_data, transform=transform)
    val_dataset = ChestXRayDataset(val_data, transform=transform)
    test_dataset = ChestXRayDataset(test_data, transform=transform)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)


    return train_loader, val_loader, test_loader


