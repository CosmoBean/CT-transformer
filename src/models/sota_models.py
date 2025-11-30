"""
State-of-the-art models for chest X-ray classification
"""
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Optional


class VisionTransformerClassifier(nn.Module):
    """
    Vision Transformer (ViT) for multi-label classification
    Based on: Dosovitskiy et al., "An Image is Worth 16x16 Words"
    """
    
    def __init__(
        self,
        num_classes: int = 15,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pretrained: bool = True,
        model_name: str = "vit_base_patch16_224",
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Use timm's ViT implementation
        if pretrained:
            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,  # Remove classifier
                in_chans=in_channels,
            )
            # Get feature dimension
            with torch.no_grad():
                dummy_input = torch.randn(1, in_channels, img_size, img_size)
                features = self.backbone(dummy_input)
                feature_dim = features.shape[1]
        else:
            # Custom ViT
            self.backbone = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                in_chans=in_channels,
            )
            with torch.no_grad():
                dummy_input = torch.randn(1, in_channels, img_size, img_size)
                features = self.backbone(dummy_input)
                feature_dim = features.shape[1]
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet for multi-label classification
    Based on: Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs"
    """
    
    def __init__(
        self,
        num_classes: int = 15,
        model_name: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Load EfficientNet backbone
        if pretrained:
            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,  # Remove classifier
            )
        else:
            self.backbone = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
            )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class ResNetClassifier(nn.Module):
    """
    ResNet for multi-label classification
    Based on: He et al., "Deep Residual Learning for Image Recognition"
    """
    
    def __init__(
        self,
        num_classes: int = 15,
        model_name: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Load ResNet backbone
        if model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
        elif model_name == "resnet152":
            self.backbone = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown ResNet model: {model_name}")
        
        # Remove final classifier
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class SwinTransformerClassifier(nn.Module):
    """
    Swin Transformer for multi-label classification
    Based on: Liu et al., "Swin Transformer: Hierarchical Vision Transformer"
    """
    
    def __init__(
        self,
        num_classes: int = 15,
        model_name: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Load Swin Transformer backbone
        if pretrained:
            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,  # Remove classifier
            )
        else:
            self.backbone = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
            )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

