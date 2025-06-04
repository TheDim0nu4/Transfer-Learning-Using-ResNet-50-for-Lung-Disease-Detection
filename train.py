from torch import nn
from torch import optim
from torchvision import models
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from class_weights import compute_pos_weights




def train( train_loader, val_loader, device ):

    output_size = 15

    resnet50 = models.resnet50( weights="IMAGENET1K_V1" )
    num_features = resnet50.fc.in_features
    resnet50.fc = nn.Sequential( nn.Dropout(0.5), nn.Linear(num_features, output_size) )

    resnet50 = resnet50.to(device)


    pos_weight = compute_pos_weights(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    learning_rate = 0.0001
    optimizer = optim.Adam(resnet50.parameters(), lr=learning_rate)

    num_epochs = 5


    auc_train, auc_validation = [], []


    for epoch in range(num_epochs):

        print(f"Epoch {epoch+1}/{num_epochs}")
    
        resnet50.train()
        all_labels, all_preds = [], []

        total_batches = len(train_loader)

        #for batch in train_loader:   
        for i, batch in enumerate(train_loader):
            print(f"Training batch {i+1}/{total_batches}", end='\r')           

            images, labels = batch
            images, labels = images.to(device), labels.to(device)
 

            optimizer.zero_grad()

            outputs = resnet50(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()


            all_labels.append(labels.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).cpu().detach().numpy())




        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)

        auc = roc_auc_score(all_labels, all_preds, average='macro')


        print(f"Train. AUC-ROC: {auc}")
        auc_train.append( auc )


  
        resnet50.eval()  
        val_labels, val_preds = [], []


        with torch.no_grad(): 

            for batch in val_loader:

                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                outputs = resnet50(images)

                val_labels.append(labels.cpu().numpy())
                val_preds.append(torch.sigmoid(outputs).cpu().detach().numpy())


        val_labels = np.concatenate(val_labels, axis=0)
        val_preds = np.concatenate(val_preds, axis=0)

        val_auc = roc_auc_score(val_labels,  val_preds,  average='macro')


        print(f"Validation. AUC-ROC: {val_auc}.\n\n")
        auc_validation.append( val_auc )


    torch.save(resnet50, "resnet50_full_model.pth")

    with open("training_result/auc_train.txt", 'w') as f:
        f.write(f"{auc_train}\n")

    with open("training_result/auc_validation.txt", 'w') as f:
        f.write(f"{auc_validation}\n")


    return resnet50


