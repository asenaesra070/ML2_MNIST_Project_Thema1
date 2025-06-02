import matplotlib.pyplot as plt

# Beispielhafte Verlustwerte pro Epoche (aus dem Training)
loss_history = [0.28, 0.10, 0.07, 0.05, 0.03]

# Epoche-Zähler erstellen
epochen = range(1, len(loss_history) + 1)

# Plot erstellen
plt.figure(figsize=(8, 5))
plt.plot(epochen, loss_history, marker='o', linestyle='-', color='b', label='Trainingsverlust')
plt.title("MLP Modell – Trainingsverlust pro Epoche")
plt.xlabel("Epoche")
plt.ylabel("Verlust (Loss)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Plot als Bild speichern
plt.savefig("mlp_loss_plot.png")

# Plot anzeigen
plt.show()
