import torch
from torchvision import datasets, transforms

def get_fashionmnist_dataloaders (batch_size=64):                                                    # 64 ist eine sehr gute Batchgröße entsprechend den gegebenen Datensätzen: Sowohl schnell als auch computerfreundlich (In der Praxis werden häufig Werte wie 32, 64, 128 verwendet.)
    transform = transforms.ToTensor()                                                               # Bilder werden in Tensoren umgewandelt ; feature scaling (0–255 → 0–1)

    train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)     # Wir packen die Daten in kleine Portionen ("Batches")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)      # Wir packen die Daten in kleine Portionen ("Batches")

    return train_loader, test_loader
