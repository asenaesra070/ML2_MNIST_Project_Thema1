"""Die .pth-Datei in den Ergebnissen als Modell wird nicht als Python-Modul (.py) importiert.
 Die .pth-Datei wird mit der Funktion torch.load() von PyTorch geladen.
 Es wird dann als state_dict in das Modell geladen."""
import torch
from models.mlp_fashion_model import FashionMNIST_MLP

model = FashionMNIST_MLP()
# Dabei werden die trainierten Gewichte aus der Datei „results/mlp_fashion_model.pth“ in dieses test Modell geladen.
model.load_state_dict(torch.load('../results/mlp_fashion_model.pth'))
# Das Modell macht lediglich Vorhersagen. Die Gewichte bleiben konstant.
model.eval()

print("Model aufgelanden")

