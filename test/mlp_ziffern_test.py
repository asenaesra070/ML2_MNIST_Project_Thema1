import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Gerät auswählen (GPU falls verfügbar, sonst CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST-Testdaten laden
transform = transforms.ToTensor()
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=1000)

# Modellstruktur wiederherstellen (muss mit dem Trainingsmodell übereinstimmen)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),               # 28x28 → 784
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)         # 10 Klassen für Ziffern 0–9
        )

    def forward(self, x):
        return self.model(x)

# Modell instanziieren und auf das Gerät verschieben
model = MLP().to(device)

# Modellparameter laden (Pfad ggf. anpassen)
model.load_state_dict(torch.load("../mlp_ziffern_model.pt", map_location=device, weights_only=False))
model.eval()
print("✅ Modell erfolgreich geladen!")

# Genauigkeit auf den Testdaten berechnen
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f" Testgenauigkeit: {accuracy:.2f}%")
