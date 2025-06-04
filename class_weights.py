import torch




def compute_pos_weights(train_loader, device):

    num_classes = 15

    pos_counts = torch.zeros(num_classes, device=device)
    neg_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():

        for _, labels in train_loader:

            labels = labels.to(device)  
            pos_counts += labels.sum(dim=0)  
            neg_counts += (1 - labels).sum(dim=0)  


    epsilon = 1e-7  
    pos_weight = neg_counts / (pos_counts + epsilon)


    return pos_weight
