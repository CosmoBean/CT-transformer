"""
Simple anomaly detection models.
"""
import torch
import torch.nn as nn
from typing import Tuple


class Autoencoder(nn.Module):
    """
    Convolutional Autoencoder for anomaly detection.
    Trained only on normal samples, high reconstruction error indicates anomaly.
    """
    
    def __init__(
        self,
        input_size: int = 224,
        latent_dim: int = 128,
        channels: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 112x112 -> 56x56
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 56x56 -> 28x28
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 28x28 -> 14x14
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Calculate encoder output size dynamically
        # After 4 conv layers with stride=2: input_size / 16
        self.encoder_output_size = input_size // 16
        encoder_output_dim = 512 * self.encoder_output_size * self.encoder_output_size
        
        # Latent space
        self.fc1 = nn.Linear(encoder_output_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, encoder_output_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            # 14x14 -> 28x28
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 28x28 -> 56x56
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 56x56 -> 112x112
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 112x112 -> 224x224
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.fc1(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.fc2(z)
        z = z.view(z.size(0), 512, self.encoder_output_size, self.encoder_output_size)
        x_recon = self.decoder(z)
        return x_recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) for anomaly detection.
    Uses KL divergence to learn a smooth latent space.
    """
    
    def __init__(
        self,
        input_size: int = 224,
        latent_dim: int = 128,
        channels: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder (same as Autoencoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Calculate encoder output size dynamically
        # After 4 conv layers with stride=2: input_size / 16
        self.encoder_output_size = input_size // 16
        encoder_output_dim = 512 * self.encoder_output_size * self.encoder_output_size
        
        # VAE specific: mean and logvar
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, encoder_output_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.fc_decode(z)
        z = z.view(z.size(0), 512, self.encoder_output_size, self.encoder_output_size)
        x_recon = self.decoder(z)
        return x_recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
