import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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
os.makedirs("../results", exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), gen_epoch_avg, label="Generator Loss", color='blue')
plt.plot(range(1, epochs + 1), dis_epoch_avg, label="Discriminator Loss", color='red')
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.title("Generator & Discriminator Verlust pro Epoche")
plt.legend()
plt.grid(True)
plt.savefig("../results/generator_loss_plot.png")
plt.tight_layout()
plt.show()
# Load saved losses from training
gen_losses = np.load("../results/generator_loss.npy")
dis_losses = np.load("../results/discriminator_loss.npy")

# Generator-Loss als Streudiagramm
plt.figure(figsize=(10, 4))
plt.plot(dis_losses, linewidth=0.3, alpha=0.6)
plt.title("Verlustdiagramm während des Trainings des Diskriminators")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("../results/discriminator_loss_plot.png")
plt.show()

# Discriminator-Loss als Streudiagramm
plt.figure(figsize=(10, 4))
plt.plot(gen_losses, linewidth=0.3, alpha=0.6, color="steelblue")
plt.title("Verlustdiagramm während des Trainings des Generators")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("../results/generator_loss_plot.png")
plt.show()

#------------------------- GAN LOSS Plot image da unter results-------------------------
#os.makedirs("../results", exist_ok=True)
#plt.savefig("../results/gan_loss_plot.png")
#plt.show()


#GAN Confusion Matrix-Bild auf dem Bildschirm anzeigen
img = mpimg.imread("../results/gan_confusion_matrix.png")
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis('off')
plt.title("GAN → MLP Klassifikation Confusion Matrix")
plt.show()
