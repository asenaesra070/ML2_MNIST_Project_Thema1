import torch                                                        # Tensoren, GPU-Unterstützung, Autograd
import numpy as np
import torch.nn as nn                                               # Modellarchitektur & Verlustfunktionen
import torch.optim as optim                                         # Optimierer für Gewichtsanpassung
from torchvision.utils import save_image                            # Visualisierung der Ergebnisse
from models.mlp_fashion_model import FashionMNIST_MLP  # Eğitilmiş sınıflandırıcıyı kullan
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# ------------- Eigene Module -------------------------
from data.data_loader import get_fashionmnist_dataloaders
from models.gan_fashion_model import Generator,Discriminator
import os                                                           # Automatische Ordnerverwaltung

# -------------------------------------HYPERPARAMETERS -----------------------------------------------------
latent_dim = 100                    # Eingabegröße für Generator (Rauschen) und z ∈ ℝ¹⁰⁰
lr = 0.0002                         # Lernrate
batch_size = 64                     # Batchgröße
epochs = 10                       # Anzahl der Epochen

# -------------------------------------DATEN LADEN ---------------------------------------------------------
train_loader, _ = get_fashionmnist_dataloaders(batch_size)

# -------------------------------------MODELLE INITIALISIEREN ----------------------------------------------
# Gerät festlegen (GPU, falls vorhanden, sonst CPU) und also GPU ist schneller als CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Mein Gerät: {device}")

#------------------------------Einführung unseres Modells und Senden an unser Gerät (GPU/CPU)-------------
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
# ---------- VERLUSTFUNKTION ------------------------------------------------------------------------------
# Binary Cross Entropy für binäre Klassifikation
criterion = nn.BCELoss()

# ---------- OPTIMIERER ------------------------
# Die betas-Werte (0.5, 0.999) sorgen für stabileres Training in GANs, besonders durch reduzierte Momentum-Effekte.
optimizer_Gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_Dis = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# ---------- OUTPUT FOLDER DÜR WAS GENERATED ---------------------------------------------------------------
os.makedirs("gan_fashion_images", exist_ok=True)

# ----------- TRAININGSSCHLEIFE -----------------------------------------------------------------------------
gen_losses = []
dis_losses = []
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(train_loader):

        # Echte Bilder vorbereiten
        real_imgs = imgs.to(device)
        valid = torch.ones(imgs.size(0), 1, device=device)    # Label 1 = echt
        fake = torch.zeros(imgs.size(0), 1, device=device)    # Label 0 = fake

        # ------------------------Trainiere Generator------------------------

        # setzt die Gradienten vom Generator auf 0 zurück (damit sich alte Gradienten nicht ansammeln)
        optimizer_Gen.zero_grad()

        # Format (Batchgröße, 100, 1, 1) → 100 = latent_dim
        z = torch.randn(imgs.size(0), latent_dim, 1, 1, device=device)   # Rauschen ist z
        gen_imgs = generator(z)                                          # Fake Bilder aber vllt  ähnlich echte Bild
        g_loss = criterion(discriminator(gen_imgs), valid)               # Generator will D täuschen
        # Berechnet die Gradienten für alle Parameter des Generators
        g_loss.backward()
        optimizer_Gen.step()

        #------------------------ Trainiere Discriminator------------------------

        # setzt die Gradienten vom Discriminator auf 0 zurück (damit sich alte Gradienten nicht ansammeln)
        optimizer_Dis.zero_grad()
        # Verlust für echte Bilder → Diskriminator soll erkennen, dass sie echt sind
        real_loss = criterion(discriminator(real_imgs), valid)
        # Verlust für gefälschte Bilder → Diskriminator soll erkennen, dass sie fake sind
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        # Berechnet die Gradienten für alle Parameter des Discriminator
        d_loss.backward()
        optimizer_Dis.step()
        # Für alle loss in einer List zu speichern, um später in Plot zu nutzen
        gen_losses.append(g_loss.item())
        dis_losses.append(d_loss.item())

    # Ergebnis pro Epoche anzeigen in terminal
    print(f"[Epoch {epoch+1}/{epochs}] in Discrimminator loss: {d_loss.item():.4f} | in Generator loss: {g_loss.item():.4f}")

    # Beispielbilder speichern 5,10,15.png ...
    if (epoch + 1) % 5 == 0:
        save_image(gen_imgs.data[:25], f"gan_fashion_images/{epoch+1:03d}.png", nrow=5, normalize=True)

# Die .npy-Datei ist wie das „Gedächtnis“ des Generators und Discriminators
np.save("../results/generator_loss.npy", np.array(gen_losses))
np.save("../results/discriminator_loss.npy", np.array(dis_losses))

# 100 ist der in der GAN-Literatur am häufigsten gewählte und ausgewogenste Wert für latent_dim
# in GANs ist oft ein sehr präzises Gleichgewicht erforderlich (Generator ↔ Diskriminator).
# Ein Wert von 0,0002 wurde in Modellen wie DCGAN vorgeschlagen und hat gute Ergebnisse geliefert.
# Für GAN BATCH SIZE = 64 ausgewählt!
# Die betas-Werte (0.5, 0.999) sorgen für stabileres Training in GANs, besonders durch reduzierte Momentum-Effekte.
# enumerate gibt uns den Index (i) und die Bilder (imgs) aus dem DataLoader zurück
    # imgs wird in jeder Iteration automatisch aus dem Loader gelesen, deshalb vorher keine Definition nötig
    # _ steht für die Labels, die wir im GAN nicht brauchen (deshalb ignorieren wir sie)
# detach() verhindert, dass Gradienten zurück zum Generator fließen

# ----------------------------- CONFUSION MATRIX FÜR GAN ----------------------------------------
print("→ Starte Klassifikation der generierten Bilder durch das MLP-Modell...")

# --- Trainierter Klassifikator wird geladen----------------------------------------------
classifier = FashionMNIST_MLP().to(device)
classifier.load_state_dict(torch.load("../results/mlp_fashion_trainierte_model.pth"))
classifier.eval()

# Generieren Sie 1000 zufällige gefälschte Bilder
z = torch.randn(1000, latent_dim, 1, 1, device=device)
gen_imgs = generator(z)

#Vorhersagt für Classen hier ist wichtig label zu haben vom MLP
with torch.no_grad():
    preds = classifier(gen_imgs)
    pred_labels = torch.argmax(preds, dim=1).cpu().numpy()

#Falsch-wahr-Beschriftungen: 100 Beispiele jeder Klasse simulieren
true_labels = np.repeat(np.arange(10), 100)

# Confusion Matrix hesapla ve kaydet
confmat = confusion_matrix(true_labels, pred_labels)
class_names = ["T-Shirt", "Hose", "Pullover", "Kleid", "Mantel", "Sandale", "Hemd", "Turnschuh", "Tasche", "Stiefel"]
df_cm = pd.DataFrame(confmat, columns=class_names, index=class_names)


plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel("Vorhergesagte Klasse")
plt.ylabel("Latente Klasse (simuliert)")
plt.title("GAN-generierte Bilder klassifiziert durch MLP")
plt.tight_layout()
os.makedirs("../results", exist_ok=True)
plt.savefig("../results/gan_confusion_matrix.png")
plt.close()
