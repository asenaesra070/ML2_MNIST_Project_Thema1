# mlp_fashion_test.py
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from models.mlp_fashion_model import FashionMNIST_MLP
from data.data_loader import get_fashionmnist_dataloaders

# Modell laden
model = FashionMNIST_MLP()
model.load_state_dict(torch.load('../results/mlp_fashion_trainierte_model.pth', map_location='cpu'))
model.eval()

# Dataloader
_, test_loader = get_fashionmnist_dataloaders()

# Loss Function
criterion = CrossEntropyLoss()

# Test loss berechnen
total_loss_summe = 0.0
for images, labels in test_loader:
    outputs = model(images)
    loss = criterion(outputs, labels)
    total_loss_summe += loss.item()

test_loss = total_loss_summe / len(test_loader)

# Save loss
np.save('../results/test_loss_value.npy', np.array([test_loss]))
print(f"Dies ist der durchschnittliche Fehlerwert (Verlust) des trainierten Modells anhand der Testdaten.: {test_loss:.4f}")
