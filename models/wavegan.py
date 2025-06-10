# models/wavegan.py
import torch.nn as nn

class WaveGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, output_length=None):
        super().__init__()
        self.latent_dim = latent_dim
        # Define architecture for output_length via parameterizable upsampling if needed
        self.net = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 512, 25, 4, 11), nn.ReLU(),
            nn.ConvTranspose1d(512, 256, 25, 4, 11), nn.ReLU(),
            nn.ConvTranspose1d(256, 128, 25, 4, 11), nn.ReLU(),
            nn.ConvTranspose1d(128, 1, 25, 4, 11), nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class WaveGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 128, 25, 4, 11), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 25, 4, 11), nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 25, 4, 11), nn.LeakyReLU(0.2),
            nn.Conv1d(512, 1, 25, 4, 11)
        )

    def forward(self, x):
        return self.net(x).mean(dim=[1,2])