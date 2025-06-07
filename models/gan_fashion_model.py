# Generator und Diskriminator werden hier definiert. Diese beiden Modelle werden kontradiktorisch(adversarial) miteinander trainiert: DCGAN-Ansatz
# image = img
# accuracy = validity

import torch
import torch.nn as nn

# z: zufälliges Rauschen
# latent_dim :

# Bilddimensionen Fashion MNIST (für 1x28x28)
# img_shape = (1, 28, 28)
#---------------- GENERATOR -------------------------------------------------
# -- Der Generator erhält zufälliges Rauschen (z) und gibt ein Bild aus -----
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
            nn.ConvTranspose2d(latent_dim, 128, kernel_size = 7, stride = 1, padding=0, bias=False),    # Ausgabe [128x7x7] und es zeigt, wie viele verschiedene Filter zum Erstellen der „Feature-Map“ verwendet werden.
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # wir machen weiter FeatureMap [128, 7, 7] -> [64, 14, 14]
            nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # wir machen weiter FeatureMap [64, 14, 14] -> [1, 28, 28]
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()                                                                                   # Ausgabe normalisiert in Bereich [-1,1]
        )
        # z, zufälliges Rauschen was als Eingabe für Generator aber sie sind in der Tat "Fake"
    def forward(self, z):
        return self.model(z)                                                                            # Bild aus Rauschen generieren

#---------------- DISCRIMINATOR ------------------------------------------------
# Discriminator-Netzwerk: Erkennt, ob ein Bild echt oder künstlich ist
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Eingabe: [1, 28, 28] → [64, 14, 14]
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # [64, 14, 14] → [128, 7, 7]
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # [128, 7, 7] → [1, 1, 1]
            nn.Conv2d(128, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()                                                                            # Gibt Wahrscheinlichkeit aus für "Real"
        )
    def forward(self, img):
        validity = self.model(img).view(-1,1)                                                        # Ausgabe: Wie "Real" ist das Bild?
        return validity







