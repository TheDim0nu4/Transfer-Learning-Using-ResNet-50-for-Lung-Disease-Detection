import torch
from prepare_data import get_loaders
from train import train
from tresholds import get_optimal_thresholds
from evaluation import evaulation




if __name__ == "__main__":

    split = {"train": 0.7, "validation": 0.15, "test": 0.15}
    train_loader, val_loader, test_loader = get_loaders( batch_size=32, data_split=split )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")


    resnet50 = train( train_loader, val_loader, device )


    optimal_thresholds = get_optimal_thresholds( resnet50, val_loader, device )
    evaulation( resnet50, test_loader, optimal_thresholds, device )




