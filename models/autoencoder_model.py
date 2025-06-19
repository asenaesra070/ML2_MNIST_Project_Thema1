import torch
import torch.nn as nn



class AutoencoderModel:
    def __init__(self):
        self.autoencoder_model, self.encoder_model = self.autoencoder_build()

    def autoencoder_build(self):
        class Autoencoder(nn.Module):
            def __init__(self):
                super(Autoencoder, self).__init__()
                self.encoder = nn.Sequential(
                    #nn.Linear(784, 32),
                    #nn.Linear(784,256),
                    #nn.ReLU(),
                    #nn.Linear(256,128),
                    #nn.ReLU(),
                    #nn.Linear(128,64),
                    #nn.ReLU()
                    nn.Linear(784, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    #nn.Linear(32, 784),

                    #nn.Linear(64,128),
                    #nn.ReLU(),
                    #nn.Linear(128,256),
                    #nn.ReLU(),
                    #nn.Linear(256,784),
                    ##nn.Sigmoid()
                    #nn.Tanh()
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, 784),
                    nn.Tanh()
                )

            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x


        autoencoder_model = Autoencoder()

        encoder_model = nn.Sequential(*autoencoder_model.encoder)

        return autoencoder_model, encoder_model