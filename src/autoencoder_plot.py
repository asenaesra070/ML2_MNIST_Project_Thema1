import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict
from scipy.ndimage import maximum_filter
from PIL import Image
import os


def save_plot(fig, dataset_name, category, filename):

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # <- geht von src/ eine Ebene hoch
    save_dir = os.path.join(base_dir, "results", "FINALAutoencoder", dataset_name, category)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path)
    print(f"Plot gespeichert unter: {save_path}")


def plothistory(train_losses, val_losses, option):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_title('Model Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.show()
    if option == 0:
        save_plot(fig,dataset_name='MNIST',category='Plots',filename='autoencoder_loss_curve.png')
    else:
        save_plot(fig, dataset_name='FashionMNIST', category='Plots', filename='autoencoder_loss_curve.png')


def comparison(test_images, test_labels, DecodedImages, EncodedClassifier, encoder_model, option):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EncodedClassifier.to(device)
    encoder_model.to(device)
    n = 5  # Anzahl der Bilder
    offset = 0  # optional: um andere Bilder anzuzeigen
    fig = plt.figure(figsize=(20, 5))

    if option == 0:
        label_map = {i: str(i) for i in range(10)}
    else:
        label_map = {
            0: "T-Shirt/Top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot"
        }

    for i in range(n):
        index = i + offset

        # Originalbild
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_images[index], cmap='gray')

        # Label über dem Bild einfügen
        plt.title(f"Original: {label_map[test_labels[index]]}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Decodiertes Bild
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(DecodedImages[index].reshape(28, 28), cmap='gray', vmin=-1, vmax=1)

        # Original-Testbild holen, normalisieren, reshapen → dann ENCODEN
        original_image = (test_images[index].astype("float32") / 255.0)*2-1.
        original_image = torch.tensor(original_image.reshape(1, 784), dtype=torch.float32).to(device)
        encoded_image = encoder_model(original_image)

        # Encoded Input → Klassifikation
        predicted_label = np.argmax(EncodedClassifier(encoded_image).detach().numpy())
        original_label = test_labels[index]

        # Titel anzeigen
        plt.title(f"Predicted: {label_map[predicted_label]}")

        # Overlay-Text: Zeige vorhergesagte Zahl im Bild
        #color = 'green' if predicted_label == original_label else 'red'
        #plt.text(0.5, 0.5, f'{predicted_label}', color=color, fontsize=16, ha='center', va='center',
         #       transform=ax.transAxes)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()
    if option == 0:
        save_plot(fig, dataset_name='MNIST', category='Classifier', filename='autoencoder_classifier_comparison.png')
    else:
        save_plot(fig, dataset_name='FashionMNIST', category='Classifier', filename='autoencoder_classifier_comparison.png')

def confusion_matrix(classifier, encoder_model, processed_test_images, test_labels, option):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.eval()
    encoder_model.eval()
    classifier.to(device)
    encoder_model.to(device)

    if option == 0:
        label_map = {i: str(i) for i in range(10)}
    else:
        label_map = {
            0: "T-Shirt/Top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot"
        }

    test_labels_tensor = torch.tensor(test_labels).to(device)

    with torch.no_grad():
        encoded_test = encoder_model(processed_test_images.to(device))
        outputs = classifier(encoded_test)
        _, predicted = torch.max(outputs, 1)

    cm = sk_confusion_matrix(test_labels_tensor.cpu(), predicted.cpu())

    fig = plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=[label_map[i] for i in range(10)],
                     yticklabels=[label_map[i] for i in range(10)])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix - Encoded Classifier')
    plt.show()
    if option == 0:
        save_plot(fig, dataset_name='MNIST', category='Classifier', filename='autoencoder_confusion_matrix.png')
    else:
        save_plot(fig, dataset_name='FashionMNIST', category='Classifier',
                  filename='autoencoder_confusion_matrix.png')
def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def compute_average_classwise_errors(original_images, reconstructed_images, labels, image_shape=(28, 28)):
    n_classes = len(np.unique(labels))
    errors_per_class = defaultdict(list)

    for img, rec, label in zip(original_images, reconstructed_images, labels):
        img = img.reshape(image_shape)
        rec = rec.reshape(image_shape)
        err = np.abs(img - rec)
        errors_per_class[int(label)].append(err)

    avg_errors_per_class = {}
    for cls, err_imgs in errors_per_class.items():
        if len(err_imgs) == 0:
            continue
        err_stack = np.stack(err_imgs)
        avg_errors_per_class[int(cls)] = np.mean(err_stack, axis=0)

    return avg_errors_per_class


def get_top_error_regions(error_map, n_regions=3, region_size=4):
    """
    Gibt die Koordinaten der n größten Fehlerregionen zurück (z. B. für Rechteckzeichnung).
    """


    # Max-Filter zur Lokalisierung von Fehlerzentren
    max_filtered = maximum_filter(error_map, size=region_size)
    mask = (error_map == max_filtered)

    # Sortiere nach Fehlerwert
    coords = list(zip(*np.where(mask)))
    coords = sorted(coords, key=lambda c: error_map[c], reverse=True)

    return coords[:n_regions]


def analyze_common_reconstruction_failures_by_class(
    original_images, reconstructed_images, labels, option, threshold_percentile=90
):
    avg_errors = compute_average_classwise_errors(original_images, reconstructed_images, labels)
    unique_classes = sorted(avg_errors.keys())
    n_classes = len(unique_classes)

    if option == 0:
        label_map = {i: str(i) for i in range(10)}
    else:
        label_map = {
            0: "T-Shirt/Top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
            5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"
        }

    fig, axs = plt.subplots(n_classes, 2, figsize=(8, 3 * n_classes))
    if n_classes == 1:
        axs = np.expand_dims(axs, axis=0)

    for row_idx, cls in enumerate(unique_classes):
        avg_error_map = avg_errors[int(cls)]

        # Schwellenwert für hohe Fehler
        threshold = np.percentile(avg_error_map, threshold_percentile)
        error_coords = list(zip(*np.where(avg_error_map > threshold)))
        error_coords = sorted(error_coords, key=lambda c: avg_error_map[c], reverse=True)[:3]

        # Finde das "schlechteste" Bild an diesen Koordinaten
        class_indices = np.where(labels == cls)[0]
        if len(class_indices) == 0:
            continue

        max_error_score = -1
        best_idx = class_indices[0]

        for idx in class_indices:
            rec_img = reconstructed_images[idx].reshape(28, 28)
            orig_img = original_images[idx].reshape(28, 28)
            err_map = np.abs(orig_img - rec_img)

            # Fehler nur an interessanten Stellen
            region_error_sum = sum(err_map[y, x] for (y, x) in error_coords if 0 <= y < 28 and 0 <= x < 28)

            if region_error_sum > max_error_score:
                max_error_score = region_error_sum
                best_idx = idx

        # Final ausgewähltes Bild
        orig_img = original_images[best_idx].reshape(28, 28)
        rec_img = reconstructed_images[best_idx].reshape(28, 28)

        # Originalbild
        ax_orig = axs[row_idx, 0]
        ax_orig.imshow(orig_img, cmap='gray', vmin=-1, vmax=1)
        ax_orig.set_title(f"Original: {label_map[int(cls)]}")
        ax_orig.axis('off')
        for (y, x) in error_coords:
            rect = patches.Rectangle((x - 2, y - 2), 4, 4, linewidth=1.5, edgecolor='red', facecolor='none')
            ax_orig.add_patch(rect)

        # Rekonstruiertes Bild
        ax_rec = axs[row_idx, 1]
        ax_rec.imshow(rec_img, cmap='gray', vmin=-1, vmax=1)
        ax_rec.set_title(f"Rekonstruktion: {label_map[int(cls)]}")
        ax_rec.axis('off')
        for (y, x) in error_coords:
            rect = patches.Rectangle((x - 2, y - 2), 4, 4, linewidth=1.5, edgecolor='red', facecolor='none')
            ax_rec.add_patch(rect)

    plt.tight_layout()

    if option == 0:
        save_plot(fig, dataset_name='MNIST', category='Plots', filename='autoencoder_error_areas.png')
    else:
        save_plot(fig, dataset_name='FashionMNIST', category='Plots',
                  filename='autoencoder_error_areas.png')
    plt.show()


def plot_class_reconstruction_with_error_regions(orig_image, rec_image, error_map, error_regions, class_label, option):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    if option == 0:
        label_map = {i: str(i) for i in range(10)}
    else:
        label_map = {
            0: "T-Shirt/Top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle Boot"
        }

    for ax, img, title in zip(axs, [orig_image, rec_image], ['Original', 'Rekonstruktion']):
        #if img.max() > 1:
         #   img_norm = img / 255.0
        #else:
         #   img_norm = img
        ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
        ax.set_title(f"{title} (Klasse: {label_map[class_label]})")
        ax.axis('off')
        for (y, x) in error_regions:
            rect = patches.Rectangle((x - 2, y - 2), 4, 4, linewidth=1.5, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
    plt.tight_layout()
    plt.show()


def save_model(model, dataset_name, category, filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, "results", "FINALAutoencoder", dataset_name, category)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Modell gespeichert unter: {save_path}")


def save_decoded_images(decoded_images, dataset_name, category='Decoded Images'):
    # Pfad zum Ordner "results/<dataset_name>/<category>"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # zwei Ebenen hoch
    save_dir = os.path.join(base_dir, "results", "FINALAutoencoder", dataset_name, category)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Speichere Bilder nach: {save_dir}")

    for i, img in enumerate(decoded_images):
        # Tensor zu NumPy, falls nötig
        if hasattr(img, 'cpu'):
            img = img.cpu().numpy()

        # Falls Bild geflattet ist (784,), in (28,28) umwandeln
        if img.ndim == 1 and img.shape[0] == 784:
            img = img.reshape(28, 28)

        # Normieren auf [0, 1]
        img_min, img_max = img.min(), img.max()
        img_norm = (img - img_min) / (img_max - img_min + 1e-8)

        # Umwandeln in 8-bit Bild
        img_uint8 = (img_norm * 255).astype(np.uint8)

        # Speichern
        save_path = os.path.join(save_dir, f"decoded_img_{i}.png")
        Image.fromarray(img_uint8).save(save_path)

    print(f"{len(decoded_images)} rekonstruierte Bilder gespeichert.")

