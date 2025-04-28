# Der Ordner models/ ist extra dafür gedacht, alle Modelldefinitionen aufzubewahren.
import torch
import torch.nn as nn
import torch.nn.functional as Fu   # Es enthält direkte Funktionen wie Aktivierungsfunktionen, Verlustfunktionen, Operationen usw.


# MLP-Modell
class FashionMNIST_MLP(nn.Module):  # nn. Module ist die Hauptklasse, auf der alle neuronalen Netzwerkmodelle in PyTorch basieren./ inheritance
    def __init__(self):
        # Die Funktion super() ruft die übergeordnete (übergeordnete) Klasse auf. Also hier: Wir führen die Funktion __init__() der Klasse nn.Module aus.
        super(FashionMNIST_MLP, self).__init__()

        # (28X 28 Pixels) = 784 Eingaben → 128 sei „üblicherweise guter Startwert“. Dieser Wert soll eine Überanpassung(overfitting) des Modells verhindern.
        self.fc1 = nn.Linear(784, 128)
        #  Zweite Schicht: 128 → 64 Neuronen
        # 32 könnte dies auch tun, aber wenn es zu klein ist, gehen wichtige Informationen verloren. Wenn es zu groß ist, gibt es zu viele Parameter → Überanpassung(overfitting).
        self.fc2 = nn.Linear(128, 64)
        # Ausgangsschicht: 64 → 10 Neuronen (Denn es gibt 10 Kurse bei FashionMNIST!)
        self.fc3 = nn.Linear(64, 10)


    # def Forward definiert den Übergang des Modells von der Eingabe zur Ausgabe. Mit Eingabe "x"
    def forward(self, x):
        # Flatten: MLP (Multilayer Perceptron) funktioniert nur mit Vektoren (eindimensionale Listen) damm flatten wir bild zu vektor!
        # Das Ziel hier: (64, 1, 28, 28) → (64, 784) zu machen
        x = x.view(x.size(0),-1)
        # RELU → Macht negative Werte gleich Null und lässt positive Werte unverändert. Model trainiert besser und bring non-linearity
        x = Fu.relu(self.fc1(x))
        x = Fu.relu(self.fc2(x))

        x = self.fc3(x)
        return x

# Modell instanziieren
model = FashionMNIST_MLP()
# Modellstruktur im Console
print(model)

# bias = True : Der Bias verschiebt die Aktivierungsfunktion nach oben oder unten! Dadurch kann das Neuron auch dann aktiv sein, wenn die Eingaben klein oder 0 sind.