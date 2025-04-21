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



📊 Daten & Vorbereitung
Als Entwickler:in möchte ich den Fashion-MNIST-Datensatz laden,
damit ich Trainings- und Testdaten nutzen kann.

Als Entwickler:in möchte ich einen DataLoader bauen,
damit ich die Daten effizient in Batches verarbeiten kann.

🧠 Modelle entwickeln
Als Entwickler:in möchte ich ein MLP-Modell bauen,
damit ich eine Basis-Klassifikation implementieren kann.

Als Entwickler:in möchte ich einen Autoencoder entwickeln,
damit ich Features extrahieren und zum Vergleich heranziehen kann.

Als Entwickler:in möchte ich ein GAN-Modell trainieren,
damit ich synthetische Bilder erzeugen und deren Klassifikation testen kann.

🧪 Training & Evaluation
Als Entwickler:in möchte ich meine Modelle trainieren und speichern,
damit ich sie später wiederverwenden oder vergleichen kann.

Als Entwickler:in möchte ich Trainingsmetriken (Loss, Accuracy etc.) ausgeben,
damit ich die Lernkurven analysieren kann.

Als Entwickler:in möchte ich eine Confusion Matrix generieren,
damit ich Fehlerquellen besser verstehe.

🔍 Vergleich & Analyse
Als Entwickler:in möchte ich die Performanz aller Modelle vergleichen,
damit ich entscheiden kann, welches Modell am besten geeignet ist.

Als Entwickler:in möchte ich untersuchen, wie gut GAN-Bilder klassifiziert werden,
damit ich den Nutzen von generierten Daten analysieren kann.

📝 Dokumentation & Präsentation
Als Teammitglied möchte ich Ergebnisse visualisieren (Plots, Tabellen),
damit die Resultate verständlich dargestellt sind.

Als Student:in möchte ich eine finale Präsentation und Dokumentation erstellen,
damit ich meine Arbeit strukturierter abgeben kann.
