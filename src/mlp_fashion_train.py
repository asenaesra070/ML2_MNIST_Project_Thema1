# mlp_fashion_train.py – Trainingslogik für AE vs GAN Modelle
import torch
import torch.nn as nn
import torch.optim as optim                                 # Mit optim können wir Modellparameter updaten (Training!)
import os
import numpy as np
from models.mlp_fashion_model import FashionMNIST_MLP       # Modell importieren
from data.data_loader import get_fashionmnist_dataloaders   # Dataloader/ get_fashionista_dataloaders importieren

# Gerät festlegen (GPU, falls vorhanden, sonst CPU) und also GPU ist schneller als CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Mein Gerät: {device}")

# Wir senden das Modell an das Gerät
model = FashionMNIST_MLP().to(device)
print(f"Wo ist mlp_fashion_model im Gerät?", next(model.parameters()).device)

# Bestimmt, wie oft das Modell den gesamten Trainingsdatensatz von Anfang bis Ende durchsucht. 10,20,100 … aber so viel macht overfitting
number_epochs = 10

# Trainings- und Test-Daten vom data_loader laden
train_loader, test_loader = get_fashionmnist_dataloaders()

"""Der Optimierer dient dazu, das Modell weights (die Gewichte) zu verbessern, also lernfähig zu machen. 
Adam Optimizer (Adaptive Moment): ist ein Optimierungsalgorithmus: Es bietet ein stabileres und schnelleres 
Training durch automatische Anpassung der Lernrate."""

optimizer = optim.Adam(model.parameters(), lr=1e-3)

"""CrossEntropyLoss ist eine Verlustfunktion, die bei Klassifizierungsproblemen verwendet wird.
Es misst den Unterschied zwischen den Vorhersagen des Modells (Ausgabe-Logits) und den tatsächlichen Labels.
Wir erwarten klein loss wert"""

criterion = nn.CrossEntropyLoss()

# ----------- TRAININGSSCHLEIFE-------------------------------
loss_values_train = []                                          # Leere Liste zur Aufnahme epochen basierter Trainings -Verlustwerte
for epochs in range (number_epochs):
    model.train()                                               # Modell in Trainingsmodus.
    running_loss_summe = 0.0                                    # Zur Berechnung des durchschnittlichen Verlustes wird hier der Gesamtverlust über mehrere Epochen hinweg erfasst.

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)   # Auf Gerät laden - CPU
        # print(images.device, labels.device)
        #  Gradienten zurücksetzen
        optimizer.zero_grad()
        # Batch daten im Modell
        outputs = model(images)
        # Loss Berechnen
        loss = criterion(outputs, labels)                       # criterion = nn.CrossEntropyLoss() und die Differenz (den Verlust) zwischen der Vorhersage des Modells (outputs) und den richtigen Antworten (labels).
        # Backpropagation
        loss.backward()
        # Gradient descent step!
        optimizer.step()
        # Wir addieren den Verlustwert dieser Charge (.item) zur Variable running_loss.
        running_loss_summe += loss.item()
    loss_values_train.append(running_loss_summe / len(train_loader))   # loss_values_train wird summiert in jede epochs
    print(f"Epoch [{epochs + 1}/{number_epochs}], Loss: {running_loss_summe / len(train_loader):.4f}")
print("Training finished")

# ---------------- TESTSCHLEIFE VERLUST MESSEN PRO EPOCH ----------------
loss_values_test = []                                           # Leere Liste zur Aufnahme epochen basierter Test -Verlustwerte
for epoch in range(number_epochs):
    model.train()
    running_loss_summe = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss_summe += loss.item()

    loss_values_train.append(running_loss_summe / len(train_loader))
    print(f"Train Loss [{epoch+1}/{number_epochs}]: {running_loss_summe / len(train_loader):.4f}")

    # Modell im Evaluierungsmodus versetzen
    model.eval()
    total_loss_test = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss_test += loss.item()

    avg_test_loss = total_loss_test / len(test_loader)
    loss_values_test.append(avg_test_loss)
    print(f"Test Loss [{epoch+1}/{number_epochs}]: {avg_test_loss:.4f}")

print("Training und Evaluation abgeschlossen.")

# ---------------- TESTVERLUST-LISTE SPEICHERN ----------------

# Wir speichern die Verlustwerte des Testdatensatzes als .npy-Datei
os.makedirs('../results', exist_ok=True)
np.save('../results/loss_values_test.npy', np.array(loss_values_test))
print("Testverlust-Liste gespeichert unter ../results/loss_values_test.npy")




# -------MODELL SPEICHERN FÜR TRAININGS-------------
# Die .pth-Datei ist wie das „Gedächtnis“ des Modells → Hier wird die trainierte Version des Modells gespeichert.
os.makedirs(name='../results', exist_ok=True)
torch.save(model.state_dict(), '../results/mlp_fashion_trainierte_model.pth')
print("Modell wurde gespeichert mit ihre Gewichte unter results/mlp_fashion_trainierte_model.pth")

# Die .npy-Datei ist wie das „Gedächtnis“ des Modells → Hier wird die trainierte verllusst mit numerics des Modells gespeichert.
save_path = "../results/loss_values.npy"
np.save(save_path, loss_values_train)
print(f"Trainingsverlust-Liste : {save_path}")
