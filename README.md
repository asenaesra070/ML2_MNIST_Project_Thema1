# ML2_MNIST_Project_Thema1
Projektarbeit für ML II: AE vs GAN mit MNIST/Fashion-MNIST




📅 21.04.2025 – Erste Struktur & Datenprüfung

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


