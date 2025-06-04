from torchvision import transforms
import random




def augment_classes(train_data, augmentation_dict):


    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=10, translate=(0.06, 0.06)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ])


    augmented_data = []


    for item in train_data:

        label_class = item['labels']  

        label_for_augmented = [ l for l in label_class if l in augmentation_dict.keys() ]

        if len(label_for_augmented) > 0:

            augment_count = max( [ augmentation_dict[l] for l in label_for_augmented] )

            augmented_data.append(item)

            for _ in range(augment_count - 1):  

                augmented_image = transform(item['image'])  
                augmented_data.append({'image': augmented_image, 'labels': label_class})

        else:
            augmented_data.append(item)


    random.shuffle(augmented_data)


    return augmented_data
