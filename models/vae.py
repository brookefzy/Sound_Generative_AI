# models/vae.py
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_length, latent_dim=128):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        enc_size = (input_length // 4) * 64
        self.fc_mu = nn.Linear(enc_size, latent_dim)
        self.fc_logvar = nn.Linear(enc_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, enc_size)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, input_length // 4)),
            nn.ConvTranspose1d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 4, 2, 1), nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        h2 = self.fc_decode(z)
        recon = self.decoder(h2)
        return recon, mu, logvar
