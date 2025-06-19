import torch
import torch.nn as nn

class CNNAutoencoderModel:
    def __init__(self, latent_dim=64):
        self.autoencoder_model, self.encoder_model = self.cnn_autoencoder_build(latent_dim)

    def cnn_autoencoder_build(self, latent_dim):
        class CNN_Autoencoder(nn.Module):
            def __init__(self, latent_dim):
                super(CNN_Autoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # 1×28×28 → 8×14×14
                    nn.ReLU(inplace=True),
                    nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 8×14×14 → 16×7×7
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 16×7×7 → 32×4×4
                    nn.ReLU(inplace=True),
                    nn.Flatten(),                                          # 32×4×4 = 512
                    nn.Linear(32 * 4 * 4, latent_dim)                      # 512 → latent_dim
                )

                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 32 * 4 * 4),                     # latent_dim → 512
                    nn.ReLU(inplace=True),
                    nn.Unflatten(dim=1, unflattened_size=(32, 4, 4)),     # 512 → 32×4×4
                    nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32×4×4 → 16×8×8
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),   # 16×8×8 → 8×16×16
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),    # 8×16×16 → 1×32×32
                    nn.Tanh()
                )

            def forward(self, x):
                z = self.encoder(x)
                recon = self.decoder(z)
                return recon[:, :, 2:30, 2:30]  # crop 32×32 → 28×28

        # Initialisiere FINALAutoencoder
        autoencoder_model = CNN_Autoencoder(latent_dim)

        # Extrahiere den Encoder als eigenes Modell
        #encoder_model = nn.Sequential(*list(autoencoder_model.encoder.children()))
        encoder_model = autoencoder_model.encoder
        return autoencoder_model, encoder_model




