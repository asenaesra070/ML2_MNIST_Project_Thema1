from typing import List
import torch                                                            # PyTorch-Bibliothek – für maschinelle Lernmodelle und Tensorberechnungen.
import matplotlib.pyplot as plt                                         # Wird zum grafischen Zeichnen verwendet (z. B. zum Anzeigen von images).
import numpy as np                                                      # Numerische Verfahren
from models.mlp_fashion_model import FashionMNIST_MLP                   # Modell importieren
from data.data_loader import get_fashionmnist_dataloaders               # Daten importieren
from sklearn.metrics import confusion_matrix    # Confusion Matrix = Die Klassen zeigt, die das Modell verwechselt.
import seaborn as sns
import pandas as pd

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
all_preds = []                                                     # Alle vom Modell vorhergesagten Werte werden hier gespeichert.
all_labels = []

# Hier sind die richtigen Klassenbezeichnungen hinterlegt.
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        values, predictions = torch.max(outputs, 1)
        sum += labels.size(0)
        correct += (labels == predictions).sum().item()
        # ---- Confusion Matrix------------------------
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
accuracy = 100 * correct / sum                                      # Die richtige Vorhersage rate wird als Prozentsatz berechnet.
print(f"Accuracy/Testdatensatz: {accuracy:.2f}%")

# ---- Confusion Matrix------------------------
confmat = confusion_matrix(all_labels, all_preds)                   # Confusion Matrix Berechnet
df_cm = pd.DataFrame(confmat, index=class_fashion_names, columns=class_fashion_names)       #
print("Lange der Vorhersagten Werte:", len(all_preds), " Lange des Label:", len(all_labels))
print("Confusion matrix shape:", confmat.shape)

#----- Laden und Plotten der Trainingsverlust aus einer Datei
loss_values = np.load("../results/loss_values.npy")
#-------Laden der Accuracy
save_path_acc = '../results/accuracy_value.npy'
np.save(save_path_acc, np.array([accuracy]))


# Plotten
plt.plot(loss_values)
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.title("Loss pro Epoche")
plt.grid(True)
plt.show()

# Confusion Matrix = Dies ist eine Tabelle, die die Klassen zeigt, die das Modell verwechselt.
# Zeilen: Tatsächliche Beschriftungen
# Spalten: Modellvorhersagen
# Plotten
plt.figure(figsize=(10, 8))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Fashion MNIST")
plt.xlabel("Vorhersage Label")
plt.ylabel("True Label")
plt.tight_layout()                                               # sorgt dafür, dass sich die Elemente im Plot nicht überschneiden
plt.savefig("../results/confusion_matrix.png")
plt.show()

# ---- Shirt, Pullover, Dress -------------------------------------
# ?  Data Augmentation
# ?  (Convolutional Neural Network - nein
# ?  class weights in Loss Function
# ?  Merkmalsextraktion z.B. mit AutoEncoder: WAS MACHEN WIR !
# ? Regularisierungstechniken:
