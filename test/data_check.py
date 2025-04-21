import sys
import os
# Da wir uns im Testordner befinden, fügen wir dem Importpfad einen übergeordneten Ordner (das Stammverzeichnis des Projekts) hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Wir importieren die Funktion get_dataloaders in die Datei data_loader.py
from data.data_loader import get_dataloaders
# Wir rufen Trainings- und Testdatenlader auf (batch_size=64 default)
train_loader, test_loader = get_dataloaders()
# Wir erhalten EINEN Batch vom Trainingsdatenlader
images, labels = next(iter(train_loader))
# Wir drucken Batchgrößen auf dem Bildschirm
print(images.shape, labels.shape)
