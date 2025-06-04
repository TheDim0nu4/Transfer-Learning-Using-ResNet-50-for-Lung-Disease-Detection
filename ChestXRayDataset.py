from torch.utils.data import Dataset
import torch




class ChestXRayDataset(Dataset):

    def __init__(self, data_split, transform=None):

        self.data = data_split
        self.transform = transform


    def __len__(self):

        return len(self.data)


    def __getitem__(self, idx):

        image = self.data[idx]["image"]
        labels = self.data[idx]["labels"]

        if self.transform:
            image = self.transform(image)

        label_tensor = torch.zeros(15)

        for l in labels:
            label_tensor[l] = 1.0


        return image, label_tensor
    
