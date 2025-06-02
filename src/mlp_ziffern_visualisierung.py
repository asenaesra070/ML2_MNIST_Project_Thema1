import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Gerät
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MLP-Klasse wie im Training
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

# Modell laden
model = MLP().to(device)
model.load_state_dict(torch.load("mlp_ziffern_model.pt", map_location=device, weights_only=False))
model.eval()

# Testdaten laden
transform = transforms.ToTensor()
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)

# Vorhersagen und Labels sammeln
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

# 1. Genauigkeit anzeigen
acc = accuracy_score(y_true, y_pred)
print(f"✅ Testgenauigkeit: {acc * 100:.2f}%")

# 2. Konfusionsmatrix plotten
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap="Blues", values_format='d')
plt.title("Konfusionsmatrix für MLP Ziffern")
plt.tight_layout()
plt.show()

# 3. Optional: Heatmap der Fehler
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Heatmap der Vorhersagefehler")
plt.xlabel("Vorhergesagt")
plt.ylabel("Wahr")
plt.show()
