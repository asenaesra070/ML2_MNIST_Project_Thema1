## ML2_MNIST_Project_Thema1
Projektarbeit für ML II: AE vs GAN mit MNIST/Fashion-MNIST

#Projekt Calendar

📅 21.04.2025 – Erste Struktur & Datenprüfung & Datenaufladen

* Neue Dateien:
data/data_loader.py:
Enthält die Funktion get_dataloaders(), die Fashion-MNIST-Daten mit torchvision.datasets.FashionMNIST lädt und DataLoader zurückgibt.

test/data_check.py:
Wird zur Überprüfung verwendet, ob die Daten korrekt geladen und in Chargen aufgeteilt werden können.

🔍 Ergebnis:
Training- und Testdaten wurden mit einer Batch-Größe von 64 erfolgreich geladen.

-> Ausgegebenes Format:
torch.Size([64, 1, 28, 28]) für die Bilder,
torch.Size([64]) für die Labels
→ Das bedeutet, 64 Graustufenbilder (1 Kanal) mit 28x28 Pixeln wurden korrekt geladen.


![image](https://github.com/user-attachments/assets/11cc36bc-2b7b-4b89-b39f-3da408f195c4)


📅 28.04.2025 – Modelerstellen und trainieren,testen
📅 29.04.2025 – Plotten,tabellen

![image](https://github.com/user-attachments/assets/e55ebf6e-9187-477e-9caf-21cd0f0d10e9)


📅 05.05.2025 – Confusion Matrix Lernen und damit arbeiten
![image](https://github.com/user-attachments/assets/f7a80c90-26d6-4325-9498-113e5d777780)


!.venv weg :  cd "C:\Users\guler\OneDrive - Hochschule Düsseldorf\Desktop\ML 2\ML2_MNIST_Project_Thema1"
## Modell-Gewichte anzeigen
Um die trainierten Gewichte des Modells zu sehen, führen Sie bitte folgenden Code aus:

```python
import torch
from models.mlp_fashion_model import FashionMNIST_MLP

model = FashionMNIST_MLP()
model.load_state_dict(torch.load("results/mlp_fashion_trainierte_model.pth", map_location="cpu"))
model.eval()

for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

´´´´

# The os.makedirs() method creates a directory recursively.
# While making directories if any intermediate directory is missing, os.makedirs() method will create them all 
# os.makedirs("c:/test/w3school")
```




# Zusammenfassung für Convolutional Transpose für GAN Generator/ Discriminators
📅 25.05.2025 bis 08.06.2025 

| Ebene | Zweck |
| ------------------------------------------- | --------------------------------------- |
| `latent_dim = 100` | Zufälliges Rauschen, Initialvektor |
| `ConvTranspose2d(latent_dim, 128, 7, 1, 0)` | Erstellt eine 7x7-Feature-Map mit 128 Filtern |
| `→ 64 → 1` | Vergrößert auf 14x14 und 28x28 und erreicht die tatsächliche Größe |
| `Tanh()` | Normalisiert Pixelwerte zwischen -1 und 1 |
