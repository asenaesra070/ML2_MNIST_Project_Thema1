import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from data.data_loader import get_dataMNIST, get_dataloaders
from models.autoencoder_model import AutoencoderModel
from src.autoencoder_plot import plothistory, comparison, confusion_matrix, analyze_common_reconstruction_failures_by_class, save_model, save_decoded_images
from sklearn.model_selection import train_test_split



#Option 0: Fuer die Durchfuehrung des Trainings anhand des MNIST-Datensatzes
#Option 1: Fuer die Durchfuehrung des Trainings anhand des FashionMNIST-Datensatzes
OPTION = 1

#Zur Reproduzierbarkeit des Versuchs
def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class AutoencoderTrainer:
    def __init__(self):

        autoencoder_model = AutoencoderModel()
        self.autoencoder_model, self.encoder_model = autoencoder_model.autoencoder_build()


        #Laden des jeweiligen Datensatzes durch die Methoden des Moduls 'data_loader.py'
        if OPTION == 0:
            train_data, test_data = get_dataMNIST()
        else:
            train_data, test_data = get_dataloaders()

        #Zuweisung der erhaltenen Daten als numpy
        self.train_images = train_data.data.numpy()
        self.train_labels = train_data.targets.numpy()
        self.test_images = test_data.data.numpy()
        self.test_labels = test_data.targets.numpy()


    def processing_dataset(self, val_split=0.2):
        #Gegebenfalls normalisieren auf [-1,1] und dann im Decoder tanh verwenden
        #Skalierung der Trainings- und Testbilder auf den Wertebereich [0,1]
        #train_images = self.train_images.astype("float32") / 255.0
        train_images = (self.train_images.astype("float32") / 255.0)*2-1
        train_labels = self.train_labels
        #test_images = self.test_images.astype("float32") / 255.0
        test_images = (self.test_images.astype("float32") / 255.0)*2-1
        test_labels = self.test_labels

        #Flatten der Bilder (von 28x28 Pixel auf einen 784-dimensionalen Vektor)
        train_images = train_images.reshape((len(train_images), -1))
        test_images = test_images.reshape((len(test_images), -1))

        #Splitten des Trainingsdatensatzes auf 80% Trainingsdatensatz und 20% Validierungsdatensatz
        train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=val_split, random_state=42, stratify=train_labels)

        #Speichern der Daten als Tensor
        train_images = torch.tensor(train_images, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        val_images = torch.tensor(val_images, dtype=torch.float32)
        val_labels = torch.tensor(val_labels, dtype=torch.long)

        test_images = torch.tensor(test_images, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)


        return train_images, train_labels, val_images, val_labels, test_images, test_labels

    #Gewichtsinitialisierung für das FINALAutoencoder-Training (Bei ReLU --> Kaiming/He-Gewichtsinitialisierung, beim Decoder/Sigmoid --> Xavier)
    @staticmethod
    def init_autoencoder_weights(m):
        if isinstance(m, nn.Linear):
            if isinstance(m, nn.Linear) and hasattr(m, 'weight'):
                if isinstance(m, nn.Sequential) or m.out_features == 784:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    #Gewichtsinitialisierung für das Classifier-Training (ReLU --> Kaiming/He-Gewichtsinitialisierung)
    @staticmethod
    def init_classifier_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    def autoencoder_training(self, processed_train_images, processed_val_images, processed_test_images, epochs=100, batch_size=256, lr=1e-3):
        #Gewichtsinitialisierung des Autoencoders
        self.autoencoder_model.apply(self.init_autoencoder_weights)
        #if OPTION == 0:
        #Lossfunktion: Binary Cross Entropy, da die Daten auf den Wertebereich von [0,1] skaliert und am Decoder mit Sigmoid gearbeitet wird
        #criterion = nn.BCELoss()
        #else:
        criterion = nn.MSELoss()

        #Definition des Optimierers
        optimizer = torch.optim.Adam(self.autoencoder_model.parameters(), lr=lr)

        #Erstellen der DataLoader für den FINALAutoencoder (Input = Ziel beim FINALAutoencoder), Shufflen des Trainingsdatensatzes nach jeder Epoche
        train_loader = DataLoader(TensorDataset(processed_train_images, processed_train_images), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(processed_val_images, processed_val_images), batch_size=batch_size, shuffle=False)

        train_losses, val_losses = [], []
        if OPTION == 0:
            print("\nFINALAutoencoder Training auf MNIST-Datensatz\n")
        else:
            print("\nFINALAutoencoder Training auf FashionMNIST-Datensatz\n")

        for epoch in range(epochs):
            #Versetzen des FINALAutoencoder-Modells in den Trainingsmodus
            self.autoencoder_model.train()
            running_loss = 0.0

            for data in train_loader:
                inputs, targets = data
                #Zurücksetzen der Gradienten
                optimizer.zero_grad()
                #Vorwärtspassage durch den FINALAutoencoder
                outputs = self.autoencoder_model(inputs)
                #Berechnung des Rekonstruktionsverlustes
                loss = criterion(outputs, inputs)
                #Backpropagation
                loss.backward()
                #Update der Gewichte
                optimizer.step()
                running_loss += loss.item()

            #train_loss: Durchschnittlicher Verlust
            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            # Validierung mit Validierungsdatensatz, versetzen des FINALAutoencoder in Evaluierungsmodus
            self.autoencoder_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in val_loader:
                    inputs, targets = data
                    outputs = self.autoencoder_model(inputs)
                    loss = criterion(outputs, inputs)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        #Durchlauf des trainierten FINALAutoencoder auf den Testdatensatz
        self.autoencoder_model.eval()
        with torch.no_grad():
            test_outputs = self.autoencoder_model(processed_test_images)
            test_loss = criterion(test_outputs, processed_test_images).item()

            print(f"Test Loss: {test_loss:.4f}")
        # Wo liegt der loss genau, beispielsweise aufgrund von fehlenden Knöpfen auf den Bildern von Jacken bei der Rekonstruktion etc.
        with torch.no_grad():
            decoded_images = self.autoencoder_model(processed_test_images).numpy()


        if OPTION == 0:
            dataset_name = "MNIST"
        else:
            dataset_name = "FashionMNIST"

        # Speicher sowohl kompletten FINALAutoencoder als auch den Encoder-Teil
        save_model(self.autoencoder_model, dataset_name, category="Models", filename="autoencoder.pth")
        save_model(self.autoencoder_model.encoder, dataset_name, category="Models", filename="encoder.pth")
        save_decoded_images(decoded_images, dataset_name)
        return train_losses, val_losses, test_loss, decoded_images



    def train_classifier_on_encoded_features(self, processed_train_images, processed_val_images, processed_test_images, train_labels, val_labels, test_labels):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_model = self.encoder_model.to(device)
        self.encoder_model.eval()

        #Definieren der Label-Datensätze für das Training, Validierung und das Testen
        train_labels = torch.tensor(train_labels)
        val_labels = torch.tensor(val_labels)
        test_labels = torch.tensor(test_labels)

        #Komprimieren der verarbeiteten Bilder mittels des trainierten Encoder-Modells
        with torch.no_grad():
            encoded_train = self.encoder_model(processed_train_images.to(device))
            encoded_val = self.encoder_model(processed_val_images.to(device))
            encoded_test = self.encoder_model(processed_test_images.to(device))

        #Definieren des Classifier-Modells
        classifier = nn.Sequential(
            #nn.Linear(encoded_train.shape[1], 64),
            nn.Linear(encoded_train.shape[1],128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10),
        ).to(device)

        #Gewichtsinitialisierung für den Classifier
        classifier.apply(self.init_classifier_weights)
        #Lossfunktion: Cross Entropy Loss, da wir mehrere Klassen haben
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)

        train_targets = train_labels.to(device)
        val_targets = val_labels.to(device)
        test_targets = test_labels.to(device)

        train_dataset = TensorDataset(encoded_train, train_targets)
        val_dataset = TensorDataset(encoded_val, val_targets)
        test_dataset = TensorDataset(encoded_test, test_targets)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128)
        test_loader = DataLoader(test_dataset, batch_size=128)

        train_losses, test_losses, val_losses = [], [], []
        train_accuracies, val_accuracies, test_accuracies = [], [], []
        print("\nEncoded Classifier Training\n")

        for epoch in range(40):
            classifier.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = classifier(inputs.to(device))
                loss = criterion(outputs, targets.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f"Epoch {epoch + 1}")

            classifier.eval()
            correct_train = 0
            total_train = 0
            with torch.no_grad():
                for inputs, targets in train_loader:
                    outputs = classifier(inputs.to(device))
                    _, predicted = torch.max(outputs, 1)
                    total_train += targets.size(0)
                    correct_train += (predicted == targets.to(device)).sum().item()

            train_accuracy = correct_train / total_train
            train_accuracies.append(train_accuracy)
            print(f"Trainings Loss: {avg_train_loss: .4f}, Accuracy: {train_accuracy:.4f}")

            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = classifier(inputs.to(device))
                    loss = criterion(outputs, targets.to(device))
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total_val += targets.size(0)
                    correct_val += (predicted == targets.to(device)).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            val_accuracy = correct_val / total_val
            val_accuracies.append(val_accuracy)

            print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        classifier.eval()
        correct_test = 0
        total_test = 0
        test_loss_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = classifier(inputs.to(device))
                loss = criterion(outputs, targets.to(device))
                test_loss_total += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets.to(device)).sum().item()

        avg_test_loss = test_loss_total / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracy = correct_test / total_test
        test_accuracies.append(test_accuracy)
        print(f"Test Loss: {avg_test_loss: .4f}, Test Accuracy: {test_accuracy:.4f}")

        return classifier, train_losses, val_losses, train_accuracies, val_accuracies, test_accuracies





def main():
    ae_dataset = AutoencoderTrainer()
    train_images, train_labels, val_images, val_labels, test_images, test_labels = ae_dataset.processing_dataset()
    train_losses, val_losses, test_loss, decoded_images = ae_dataset.autoencoder_training(train_images, val_images, test_images)
    plothistory(train_losses, val_losses,OPTION)
    encoded_classifier,train_losses, val_losses, train_accuracies, val_accuracies, test_accuracies = ae_dataset.train_classifier_on_encoded_features(train_images, val_images, test_images, train_labels, val_labels, test_labels)
    comparison(ae_dataset.test_images, ae_dataset.test_labels, decoded_images, encoded_classifier, ae_dataset.encoder_model, OPTION)
    confusion_matrix(encoded_classifier,ae_dataset.encoder_model, test_images, ae_dataset.test_labels, OPTION)
    analyze_common_reconstruction_failures_by_class(test_images,
    decoded_images,
    test_labels, OPTION)


if __name__ == "__main__":
    main()