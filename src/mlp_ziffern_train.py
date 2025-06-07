import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# 1. Gerät auswählen (GPU falls verfügbar, sonst CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. MNIST-Daten laden
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# 3. MLP-Modell definieren
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # 28x28 → 784
            nn.Linear(784, 128),  # 784 → 128
            nn.ReLU(),
            nn.Linear(128, 64),  # 128 → 64
            nn.ReLU(),
            nn.Linear(64, 10)  # 64 → 10 Klassen (Ziffern 0–9)
        )

    def forward(self, x):
        return self.model(x)

# 4. Modell, Loss und Optimizer vorbereiten
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training starten
loss_history = []
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoche {epoch + 1}/{epochs}, Verlust: {avg_loss:.4f}")

# 6. Verlustverlauf plotten
plt.plot(range(1, epochs + 1), loss_history, marker='o')
plt.title("Trainingsverlust pro Epoche")
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("mlp_loss_plot.png")
plt.show()

# 7. Modell speichern
import os
print(" Modell wird gespeichert im Pfad:", os.getcwd())
torch.save(model.state_dict(), "mlp_ziffern_model.pt")
print(" Modell wurde gespeichert als: mlp_ziffern_model.pt")
