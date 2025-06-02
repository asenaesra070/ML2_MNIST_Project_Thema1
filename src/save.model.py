import torch
import os
from mlp_ziffern_train import MLP  # Modellklasse importieren

# Dummy-Modell erstellen (nur Struktur)
model = MLP()

# Dateiname festlegen (ohne f-string!)
filename = "mlp_ziffern_model.pt"

# Speichern
print("📁 Modell wird gespeichert im Pfad:", os.getcwd())
torch.save(model.state_dict(), filename)
print("✅ Modell wurde gespeichert als:", filename)
