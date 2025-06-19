import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from data.data_loader import get_dataMNIST, get_dataloaders
from models.convolutional_autoencoder_model import CNNAutoencoderModel
from src.convolutional_autoencoder_plot import plothistory, comparison, confusion_matrix, visualize_single_reconstruction_torch, \
    analyze_reconstruction_dataset, analyze_common_reconstruction_failures_by_class, save_model, save_decoded_images
from sklearn.model_selection import train_test_split

# Option 0: MNIST
# Option 1: FashionMNIST
OPTION = 1

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
    def __init__(self, latent_dim=64):  # *** GEÄNDERT: latent_dim Parameter hinzugefügt ***
        # CNN FINALAutoencoder mit latent_dim
        autoencoder_model = CNNAutoencoderModel(latent_dim)  # *** GEÄNDERT ***
        self.autoencoder_model, self.encoder_model = autoencoder_model.cnn_autoencoder_build(latent_dim)

        if OPTION == 0:
            train_data, test_data = get_dataMNIST()
        else:
            train_data, test_data = get_dataloaders()

        self.train_images = train_data.data.numpy()
        self.train_labels = train_data.targets.numpy()
        self.test_images = test_data.data.numpy()
        self.test_labels = test_data.targets.numpy()

    def processing_dataset(self, val_split=0.2):
        # Bilder normalisieren auf [-1,1], da Decoder tanh benutzt
        train_images = (self.train_images.astype("float32") / 255.0) * 2 - 1
        train_labels = self.train_labels
        test_images = (self.test_images.astype("float32") / 255.0) * 2 - 1
        test_labels = self.test_labels

        # *** GEÄNDERT: keine Flatten mehr, sondern auf (N,1,28,28) reshape ***
        train_images = train_images.reshape((-1, 1, 28, 28))
        test_images = test_images.reshape((-1, 1, 28, 28))

        # Split Training / Validierung
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images, train_labels, test_size=val_split, random_state=42, stratify=train_labels
        )

        # Tensor konvertieren
        train_images = torch.tensor(train_images, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        val_images = torch.tensor(val_images, dtype=torch.float32)
        val_labels = torch.tensor(val_labels, dtype=torch.long)

        test_images = torch.tensor(test_images, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        return train_images, train_labels, val_images, val_labels, test_images, test_labels

    @staticmethod
    def init_autoencoder_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def init_classifier_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def autoencoder_training(self, processed_train_images, processed_val_images, processed_test_images, epochs=100, batch_size=256, lr=1e-3):
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        self.autoencoder_model.to(device)
        self.autoencoder_model.apply(self.init_autoencoder_weights)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.autoencoder_model.parameters(), lr=lr)

        train_loader = DataLoader(TensorDataset(processed_train_images, processed_train_images), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(processed_val_images, processed_val_images), batch_size=batch_size, shuffle=False)

        train_losses, val_losses = [], []
        print("\nFINALAutoencoder Training auf {}-Datensatz\n".format("MNIST" if OPTION == 0 else "FashionMNIST"))

        for epoch in range(epochs):
            self.autoencoder_model.train()
            running_loss = 0.0

            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.autoencoder_model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            self.autoencoder_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.autoencoder_model(inputs)
                    loss = criterion(outputs, inputs)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        self.autoencoder_model.eval()
        with torch.no_grad():
            test_outputs = self.autoencoder_model(processed_test_images)
            test_loss = criterion(test_outputs, processed_test_images).item()

            print(f"Test Loss: {test_loss:.4f}")

            # Ausgabe in numpy für spätere Visualisierung
            decoded_images = test_outputs.cpu().numpy()  # *** GEÄNDERT: .cpu() hinzugefügt für CPU Nutzung ***

        dataset_name = "MNIST" if OPTION == 0 else "FashionMNIST"
        # try-except Codeblock drum
        save_model(self.autoencoder_model, dataset_name, category="Models", filename="autoencoder.pth")
        save_model(self.encoder_model, dataset_name, category="Models", filename="encoder.pth")
        save_decoded_images(decoded_images, dataset_name)
        return train_losses, val_losses, test_loss, decoded_images, dataset_name

    def train_classifier_on_encoded_features(self, processed_train_images, processed_val_images, processed_test_images,
                                             train_labels, val_labels, test_labels):
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        self.encoder_model = self.encoder_model.to(device)
        self.encoder_model.eval()

        # Konvertiere Bilder in CNN-kompatibles Format: (N, 1, 28, 28)
        #processed_train_images = processed_train_images.view(-1, 1, 28, 28)
        #processed_val_images = processed_val_images.view(-1, 1, 28, 28)
        #processed_test_images = processed_test_images.view(-1, 1, 28, 28)

        #train_labels = torch.tensor(train_labels)
        #val_labels = torch.tensor(val_labels)
        #test_labels = torch.tensor(test_labels)
        if isinstance(train_labels, torch.Tensor):
            train_labels = train_labels.detach().clone()
        else:
            train_labels = torch.tensor(train_labels)

        if isinstance(val_labels, torch.Tensor):
            val_labels = val_labels.detach().clone()
        else:
            val_labels = torch.tensor(val_labels)

        if isinstance(test_labels, torch.Tensor):
            test_labels = test_labels.detach().clone()
        else:
            test_labels = torch.tensor(test_labels)




        with torch.no_grad():
            encoded_train = self.encoder_model(processed_train_images.to(device))
            encoded_val = self.encoder_model(processed_val_images.to(device))
            encoded_test = self.encoder_model(processed_test_images.to(device))

        # Einfacher Klassifikator
        classifier = nn.Sequential(
            nn.Linear(encoded_train.shape[1], 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10),
        ).to(device)

        classifier.apply(self.init_classifier_weights)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)

        train_dataset = TensorDataset(encoded_train, train_labels.to(device))
        val_dataset = TensorDataset(encoded_val, val_labels.to(device))
        test_dataset = TensorDataset(encoded_test, test_labels.to(device))

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128)
        test_loader = DataLoader(test_dataset, batch_size=128)

        train_losses, val_losses, test_losses = [], [], []
        train_accuracies, val_accuracies, test_accuracies = [], [], []

        print("\nEncoded Classifier Training\n")

        for epoch in range(40):
            classifier.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = classifier(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Trainingsgenauigkeit
            classifier.eval()
            correct_train = 0
            total_train = 0
            with torch.no_grad():
                for inputs, targets in train_loader:
                    outputs = classifier(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total_train += targets.size(0)
                    correct_train += (predicted == targets).sum().item()

            train_accuracy = correct_train / total_train
            train_accuracies.append(train_accuracy)

            # Validierungsgenauigkeit
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = classifier(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total_val += targets.size(0)
                    correct_val += (predicted == targets).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            val_accuracy = correct_val / total_val
            val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch + 1}")
            print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Testdaten
        classifier.eval()
        correct_test = 0
        total_test = 0
        test_loss_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = classifier(inputs)
                loss = criterion(outputs, targets)
                test_loss_total += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets).sum().item()

        avg_test_loss = test_loss_total / len(test_loader)
        test_accuracy = correct_test / total_test
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        return classifier, train_losses, val_losses, train_accuracies, val_accuracies, test_accuracies

def main():
    ae_dataset = AutoencoderTrainer(latent_dim=64)  # *** GEÄNDERT: latent_dim Parameter ***
    train_images, train_labels, val_images, val_labels, test_images, test_labels = ae_dataset.processing_dataset()
    train_losses, val_losses, test_loss, decoded_images, dataset_name = ae_dataset.autoencoder_training(train_images, val_images, test_images)
    plothistory(train_losses, val_losses, OPTION)
    encoded_classifier, train_losses, val_losses, train_accuracies, val_accuracies, test_accuracies = ae_dataset.train_classifier_on_encoded_features(
        train_images, val_images, test_images, train_labels, val_labels, test_labels)

    comparison(ae_dataset.test_images, ae_dataset.test_labels, decoded_images, encoded_classifier, ae_dataset.encoder_model, OPTION)
    confusion_matrix(encoded_classifier, ae_dataset.encoder_model, test_images, ae_dataset.test_labels, OPTION)

    analyze_common_reconstruction_failures_by_class(test_images, decoded_images, test_labels, OPTION)

if __name__ == "__main__":
    main()