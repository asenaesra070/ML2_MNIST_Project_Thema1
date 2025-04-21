import torch
from torchvision import datasets, transforms

def get_dataloaders(batch_size=64):
    transform = transforms.ToTensor()

    train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
