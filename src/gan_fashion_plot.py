import numpy as np
import matplotlib.pyplot as plt
import os

# GAN Train Loss List hinzufügen via Numpy
gen_loss = np.load("../results/generator_loss.npy")
dis_loss = np.load("../results/discriminator_loss.npy")

#Durchschnitt pro Epoche berechnen (jede Epoche kann so viele Werte wie Batchnummern haben)
epochs = 10
batches_per_epoch = len(gen_loss) // epochs
# reshape - um alle Batchverluste auf Epochenbasis zu mitteln.
gen_epoch_avg = np.mean(gen_loss.reshape(epochs, batches_per_epoch), axis=1)
dis_epoch_avg = np.mean(dis_loss.reshape(epochs, batches_per_epoch), axis=1)

#------------------------- LOSS GRAPH mit Epoch für Generator und Discriminator-------------------------
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), gen_epoch_avg, label="Generator Loss", color='blue')
plt.plot(range(1, epochs + 1), dis_epoch_avg, label="Discriminator Loss", color='red')
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.title("Generator & Discriminator Verlust pro Epoche")
plt.legend()
plt.grid(True)
plt.tight_layout()

#------------------------- GAN LOSS Plot image da unter results-------------------------
os.makedirs("../results", exist_ok=True)
plt.savefig("../results/gan_loss_plot.png")
plt.show()
