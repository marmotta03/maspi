import torch.nn as nn
import torch

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latend_dim = latent_dim
        # usando nn.Sequential definan las capas del stacked autoencoder, 
        # tanto en el encoder, como en el decoder
        self.encoder_1 = nn.Linear(in_features=784, out_features=256)
        self.encoder_2 = nn.Linear(in_features=256, out_features=128)
        self.encoder_3 = nn.Linear(in_features=128, out_features=latent_dim)
        self.decoder_1 = nn.Linear(in_features=latent_dim, out_features=128)
        self.decoder_2 = nn.Linear(in_features=128, out_features=256)
        self.decoder_3 = nn.Linear(in_features=256, out_features=784)

    def forward(self, x):
        x = x.view(len(x), -1)
        # COMPLETAR AQUI
        e_1 = torch.tanh(self.encoder_1(x))
        e_2 = torch.tanh(self.encoder_2(e_1))
        e_3 = torch.tanh(self.encoder_3(e_2))
        d_1 = torch.tanh(self.decoder_1(e_3))
        d_2 = torch.tanh(self.decoder_2(d_1))
        d_3 = torch.tanh(self.decoder_3(d_2))
        x = torch.tanh(d_3)
        ############################
        x = x.view(len(x), 1, 28, 28)
        return x