from typing import List

import torch                                                    # PyTorch-Bibliothek – für maschinelle Lernmodelle und Tensorberechnungen.
import matplotlib.pyplot as plt                                 # Wird zum grafischen Zeichnen verwendet (z. B. zum Anzeigen von images).
import numpy as np                                              # Numerische Verfahren
from models.mlp_fashion_model import FashionMNIST_MLP           # Modell importieren
from data.data_loader import get_fashionmnist_dataloaders       # Daten importieren



# --- Model Aufladen und Eval Mode ------------------------------------------------------------------
model = FashionMNIST_MLP()                                      # Model Aufladen
# Dabei werden die trainierten Gewichte aus der Datei „results/mlp_fashion_model.pth“ in dieses test Modell geladen.
model.load_state_dict(torch.load('../results/mlp_fashion_model.pth'))
model.eval()                                                    # „Evaluierungsmodus“ – in diesem Modus sind während des Trainings verwendete Techniken wie Dropout deaktiviert.
print("Model loaded")

_,test_loader = get_fashionmnist_dataloaders()                   # Test Dataloader

# ----- 10 Klassen in den FashionMNIST Datensätz (label bedeutung) - Supervised Learning-------------
class_fashion_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# ---- Die ersten 64 Bilder und deren Label aus den Testdaten. Also 1 Batch
images, labels = next(iter(test_loader))

with torch.no_grad():                                            # Die Gradienten berechnung ist deaktiviert (funktioniert schneller), da kein Training durchgeführt wird.
    outputs = model(images)
    values, predictions = torch.max(outputs, 1)                  # höchste Wahrscheinlichkeit pro Bild wähle


# ---- 9 Outputs Images mit Label werden angezeigt (FashionMNIST) , 3X3 = 9 ...
plt.figure(figsize=(10, 10))                                     # Definiert einen 10x10 großen Grafikbereich.
for i in range(16):
    plt.subplot(4, 4, i+1)                                       # 3x3 Matrix → 9 Image
    plt.imshow(images[i][0], cmap='gray')                        # (1x28x28)
    # Es schreibt sowohl das eigentliche Label als auch die predictions des Modells als Titel.
    plt.title(f"Label: {class_fashion_names[labels[i]]}\nPrediction: {class_fashion_names[predictions[i]]}")
    plt.axis("off")
plt.tight_layout()
plt.show()


# ----- Wir berechnen die Genauigkeit(Accuracy) des Modells:
correct = 0
sum = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        values, predictions = torch.max(outputs, 1)
        sum += labels.size(0)
        correct += (labels == predictions).sum().item()
accuracy = 100 * correct / sum                                      # Die richtige Vorhersagerate wird als Prozentsatz berechnet.
print(f"Accuracy/Testdatensatz: {accuracy:.2f}%")


#----- Laden und Plotten der Trainingsverlust aus einer Datei
loss_values = np.load("../results/loss_values.npy")

# Plot çiz
plt.plot(loss_values)
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.title("Loss pro Epoche")
plt.grid(True)
plt.show()