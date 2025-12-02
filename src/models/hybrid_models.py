"""
Hybrid models combining different techniques for improved performance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import timm

from .flare_model import FLAREClassifier, FLAREBlock


class CrossAttentionFusion(nn.Module):
    """Cross-attention module for fusing CNN and patch features"""
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        query: [B, N_q, embed_dim] - CNN features
        key_value: [B, N_kv, embed_dim] - Patch features
        Returns: [B, N_q, embed_dim] - Fused features
        """
        # Ensure inputs are contiguous for DataParallel compatibility
        query = query.contiguous()
        key_value = key_value.contiguous()
        
        B, N_q, _ = query.shape
        _, N_kv, _ = key_value.shape
        
        q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2).contiguous()  # [B, H, N_q, D]
        k = self.k_proj(key_value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2).contiguous()  # [B, H, N_kv, D]
        v = self.v_proj(key_value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2).contiguous()  # [B, H, N_kv, D]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N_q, self.embed_dim)
        out = self.out_proj(out)
        out = self.norm(query + out)  # Residual connection
        return out.contiguous()


class FLAREHybridClassifier(nn.Module):
    """
    Hybrid model combining FLARE transformer with CNN backbone features.
    Uses CNN for local features and FLARE for global attention with cross-attention fusion.
    """
    
    def __init__(
        self,
        num_classes: int = 15,
        img_size: int = 512,
        cnn_backbone: str = "efficientnet_b3",
        embed_dim: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        num_latents: int = 64,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # CNN backbone for local feature extraction
        self.cnn_backbone = timm.create_model(
            cnn_backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
        )
        
        # Get CNN feature dimension and spatial size
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            cnn_features = self.cnn_backbone(dummy_input)
            cnn_feat_dim = cnn_features.shape[1]
            cnn_h, cnn_w = cnn_features.shape[2], cnn_features.shape[3]
            cnn_spatial_size = cnn_h * cnn_w
        
        # Project CNN features to FLARE embedding dimension
        self.cnn_proj = nn.Sequential(
            nn.Conv2d(cnn_feat_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        
        # Patch embedding for FLARE (global structure)
        patch_size = 16
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Cross-attention fusion: CNN features query patch features
        self.cross_attn = CrossAttentionFusion(embed_dim, num_heads=num_heads, dropout=dropout)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding for CNN features (with class token)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + cnn_spatial_size, embed_dim))
        
        # FLARE blocks for processing fused features
        self.blocks = nn.ModuleList([
            FLAREBlock(
                channel_dim=embed_dim,
                num_heads=num_heads,
                num_latents=num_latents,
                act='gelu',
                rmsnorm=True,
                num_layers_kv_proj=3,
                num_layers_ffn=3,
                kv_proj_mlp_ratio=1.0,
                ffn_mlp_ratio=1.0,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head[-1].weight, std=0.02)
        if self.head[-1].bias is not None:
            nn.init.zeros_(self.head[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Ensure input is contiguous for DataParallel compatibility
        x = x.contiguous()
        
        # Extract CNN features (local patterns) - [B, embed_dim, H', W']
        cnn_features = self.cnn_backbone(x)
        cnn_features = self.cnn_proj(cnn_features)
        cnn_features = cnn_features.flatten(2).transpose(1, 2).contiguous()  # [B, N_cnn, embed_dim]
        
        # Extract patch embeddings (global structure) - [B, N_patch, embed_dim]
        patch_features = self.patch_embed(x)
        patch_features = patch_features.flatten(2).transpose(1, 2).contiguous()
        
        # Cross-attention: CNN features attend to patch features
        fused_features = self.cross_attn(cnn_features, patch_features)  # [B, N_cnn, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, fused_features), dim=1).contiguous()  # [B, 1 + N_cnn, embed_dim]
        x = x + self.pos_embed
        
        # FLARE transformer
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(x)
        
        # Use class token for classification
        out = x[:, 0].contiguous()
        out = self.head(out)
        return out


class ScaleBackbone(nn.Module):
    """FLARE backbone for a single scale"""
    def __init__(
        self,
        scale: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        num_latents: int,
    ):
        super().__init__()
        patch_size = 16
        num_patches = (scale // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            FLAREBlock(
                channel_dim=embed_dim,
                num_heads=num_heads,
                num_latents=num_latents,
                act='gelu',
                rmsnorm=True,
                num_layers_kv_proj=3,
                num_layers_ffn=3,
                kv_proj_mlp_ratio=1.0,
                ffn_mlp_ratio=1.0,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input"""
        # Ensure input is contiguous for DataParallel compatibility
        x = x.contiguous()
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, N, embed_dim] - ensure contiguous
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1).contiguous()
        x = x + self.pos_embed
        
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(x)
        
        return x[:, 0].contiguous()  # Return class token [B, embed_dim] - ensure contiguous


class MultiScaleFLARE(nn.Module):
    """
    Multi-scale FLARE that processes images at multiple resolutions.
    Uses cross-scale attention for better feature fusion.
    """
    
    def __init__(
        self,
        num_classes: int = 15,
        img_size: int = 512,
        embed_dim: int = 768,
        depth: int = 10,
        num_heads: int = 12,
        num_latents: int = 64,
        dropout: float = 0.1,
        scales: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        # Default scales: half and full resolution
        self.scales = scales if scales is not None else (img_size // 2, img_size)
        self.embed_dim = embed_dim
        
        # Create FLARE backbones for each scale
        depth_per_scale = depth // len(self.scales)
        self.scale_backbones = nn.ModuleList([
            ScaleBackbone(
                scale=scale,
                embed_dim=embed_dim,
                depth=depth_per_scale,
                num_heads=num_heads,
                num_latents=num_latents,
            )
            for scale in self.scales
        ])
        
        # Cross-scale attention: fine-scale queries coarse-scale
        self.cross_scale_attn = CrossAttentionFusion(embed_dim, num_heads=num_heads, dropout=dropout)
        
        # Feature fusion with learnable weights
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales)) / len(self.scales))
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous for DataParallel compatibility
        x = x.contiguous()
        B = x.shape[0]
        scale_features = []
        
        # Process at each scale
        for scale, backbone in zip(self.scales, self.scale_backbones):
            # Resize to scale - ensure contiguous before passing to backbone
            if scale != self.img_size:
                x_scale = F.interpolate(x, size=(scale, scale), mode='bilinear', align_corners=False)
            else:
                x_scale = x
            
            # Ensure contiguous right before passing to backbone (critical for DataParallel)
            x_scale = x_scale.contiguous()
            
            # Extract features
            feat = backbone(x_scale)  # [B, embed_dim]
            scale_features.append(feat.contiguous())  # Ensure contiguous
        
        # Cross-scale attention: fine-scale (last) queries all scales
        if len(scale_features) > 1:
            # Fine-scale features attend to coarse-scale features
            query = scale_features[-1].unsqueeze(1).contiguous()  # [B, 1, embed_dim]
            # Stack features and ensure contiguous
            key_value = torch.stack(scale_features[:-1], dim=1).contiguous()  # [B, num_scales-1, embed_dim]
            enhanced_fine = self.cross_scale_attn(query, key_value).squeeze(1).contiguous()  # [B, embed_dim]
            
            # Weighted combination of all scales - ensure contiguous
            weights = F.softmax(self.scale_weights, dim=0)
            fused = sum(w * feat.contiguous() for w, feat in zip(weights[:-1], scale_features[:-1]))
            fused = (fused + weights[-1] * enhanced_fine).contiguous()
        else:
            fused = scale_features[0].contiguous()
        
        # Classify
        out = self.head(fused)
        return out


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling for adaptive feature aggregation"""
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, embed_dim] - Patch features
        Returns: [B, embed_dim] - Pooled features
        """
        # Ensure input is contiguous for DataParallel compatibility
        x = x.contiguous()
        
        B, N, _ = x.shape
        
        # Expand learnable query
        q = self.query.expand(B, -1, -1).contiguous()  # [B, 1, embed_dim]
        q = self.q_proj(q).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()  # [B, H, 1, D]
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()  # [B, H, N, D]
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()  # [B, H, N, D]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, 1, self.embed_dim)
        out = self.out_proj(out)
        out = self.norm(out).squeeze(1).contiguous()  # [B, embed_dim]
        return out


class FLAREWithAttentionPooling(nn.Module):
    """
    FLARE with multi-head attention-based pooling instead of class token.
    Uses multi-head attention to aggregate patch features adaptively.
    """
    
    def __init__(
        self,
        num_classes: int = 15,
        img_size: int = 512,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        num_latents: int = 64,
        dropout: float = 0.1,
        pretrained: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        patch_size = 16
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # FLARE blocks
        self.blocks = nn.ModuleList([
            FLAREBlock(
                channel_dim=embed_dim,
                num_heads=num_heads,
                num_latents=num_latents,
                act='gelu',
                rmsnorm=True,
                num_layers_kv_proj=3,
                num_layers_ffn=3,
                kv_proj_mlp_ratio=1.0,
                ffn_mlp_ratio=1.0,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Multi-head attention pooling
        self.attn_pool = MultiHeadAttentionPooling(embed_dim, num_heads=num_heads, dropout=dropout)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.attn_pool.query, std=0.02)
        nn.init.trunc_normal_(self.head[-1].weight, std=0.02)
        if self.head[-1].bias is not None:
            nn.init.zeros_(self.head[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous for DataParallel compatibility
        x = x.contiguous()
        
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B, num_patches, embed_dim]
        x = x + self.pos_embed
        
        # FLARE transformer
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(x)
        
        # Multi-head attention pooling
        pooled = self.attn_pool(x)  # [B, embed_dim]
        
        # Classify
        out = self.head(pooled)
        return out

