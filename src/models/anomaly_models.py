"""
Creative anomaly detection models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import timm


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


class AnoGAN(nn.Module):
    """
    AnoGAN: Unsupervised Anomaly Detection via Adversarial Training
    Based on: Schlegl et al., "Unsupervised Anomaly Detection with GANs"
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        img_size: int = 224,
        channels: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 512 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (512, 7, 7)),
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
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
    
    def generate(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)
    
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)


class ContrastiveAnomalyDetector(nn.Module):
    """
    Contrastive Learning for Anomaly Detection
    Uses SimCLR-style contrastive learning to learn representations
    """
    
    def __init__(
        self,
        backbone_name: str = "resnet50",
        projection_dim: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()
        
        # Backbone encoder
        if pretrained:
            self.encoder = timm.create_model(
                backbone_name,
                pretrained=True,
                num_classes=0,
            )
        else:
            self.encoder = timm.create_model(
                backbone_name,
                pretrained=False,
                num_classes=0,
            )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.encoder(dummy_input)
            feature_dim = features.shape[1]
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        projections = self.projection(features)
        return F.normalize(projections, dim=1)

