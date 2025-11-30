"""
FLARE model for chest X-ray classification
FLARE uses efficient latent attention mechanism
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange


# Activation Functions
ACTIVATIONS = {
    'gelu': nn.GELU(approximate='tanh'),
    'silu': nn.SiLU(),
}


# Residual MLP Block
class ResidualMLP(nn.Module):
    def __init__(
            self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2,
            act: str = None, input_residual: bool = False, output_residual: bool = False,
        ):
        super().__init__()

        self.num_layers = num_layers
        assert self.num_layers >= -1, f"num_layers must be at least -1. Got {self.num_layers}."

        if self.num_layers == -1:
            self.fc = nn.Linear(in_dim, out_dim)
            self.residual = input_residual and output_residual and (in_dim == out_dim)
            return

        self.act = ACTIVATIONS[act] if act else ACTIVATIONS['gelu']
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fcs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.input_residual  = input_residual  and (in_dim  == hidden_dim)
        self.output_residual = output_residual and (hidden_dim == out_dim)

    def forward(self, x):
        if self.num_layers == -1:
            x = x + self.fc(x) if self.residual else self.fc(x)
            return x

        x = x + self.act(self.fc1(x)) if self.input_residual else self.act(self.fc1(x))
        for fc in self.fcs:
            x = x + self.act(fc(x))
        x = x + self.fc2(x) if self.output_residual else self.fc2(x)
        return x


# FLARE Attention
class FLARE(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = 8,
        num_latents: int = 32,
        attn_scale: float = 1.0,
        act: str = None,
        num_layers_kv_proj: int = 3,
        kv_proj_mlp_ratio: float = 1.0,
    ):
        super().__init__()

        self.channel_dim = channel_dim
        self.num_latents = num_latents
        self.num_heads = channel_dim // 8 if num_heads is None else num_heads
        self.head_dim = self.channel_dim // self.num_heads

        assert self.channel_dim % self.num_heads == 0, f"channel_dim must be divisible by num_heads. Got {self.channel_dim} and {self.num_heads}."
        assert attn_scale > 0.0, f"attn_scale must be greater than 0. Got {attn_scale}."

        # Attention scale: PyTorch's scaled_dot_product_attention uses 1/sqrt(d) by default
        # If attn_scale=1.0, we should use None to get default behavior, or compute 1/sqrt(d)
        # However, FLARE original code uses scale=1.0 explicitly, which overrides default
        # For training stability, we use proper scaling: 1/sqrt(head_dim)
        self.attn_scale = attn_scale if attn_scale != 1.0 else (1.0 / (self.head_dim ** 0.5))

        self.latent_q = nn.Parameter(torch.empty(self.channel_dim, self.num_latents))
        nn.init.normal_(self.latent_q, mean=0.0, std=0.1)

        self.k_proj, self.v_proj = [
            ResidualMLP(
                in_dim=self.channel_dim,
                hidden_dim=int(self.channel_dim * kv_proj_mlp_ratio),
                out_dim=self.channel_dim,
                num_layers=num_layers_kv_proj,
                act=act,
                input_residual=True,
                output_residual=True,
            ) for _ in range(2)
        ]

        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

    def forward(self, x, return_scores: bool = False):
        # x: [B N C]
        q = self.latent_q.view(self.num_heads, self.num_latents, self.head_dim) # [H M D]
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads) # [B H N D]
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)

        if not return_scores:
            q = q.unsqueeze(0).expand(x.size(0), -1, -1, -1) # required for fused attention
            # Use proper attention scaling for training stability
            # PyTorch's scaled_dot_product_attention uses 1/sqrt(d) by default
            # We explicitly pass the scale to ensure correct behavior
            z = F.scaled_dot_product_attention(q, k, v, scale=self.attn_scale)
            y = F.scaled_dot_product_attention(k, q, z, scale=self.attn_scale)
            scores = None
        else:
            # Manual computation with proper scaling
            scores = (q @ k.transpose(-2, -1)) * self.attn_scale  # [B H M N]
            W_encode = F.softmax(scores, dim=-1)
            W_decode = F.softmax(scores.transpose(-2, -1), dim=-1)
            z = W_encode @ v # [B H M D]
            y = W_decode @ z # [B H N D]

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.out_proj(y)
        return y, scores


# FLARE Block
class FLAREBlock(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        num_heads: int = None,
        num_latents: int = None,
        attn_scale: float = 1.0,
        act: str = None,
        rmsnorm: bool = False,
        num_layers_kv_proj: int = 3,
        num_layers_ffn: int = 3,
        kv_proj_mlp_ratio: float = 1.0,
        ffn_mlp_ratio: float = 1.0,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.norm2 = nn.RMSNorm(channel_dim) if rmsnorm else nn.LayerNorm(channel_dim)
        self.att = FLARE(
            channel_dim=channel_dim,
            num_heads=num_heads,
            num_latents=num_latents,
            attn_scale=attn_scale,
            act=act,
            num_layers_kv_proj=num_layers_kv_proj,
            kv_proj_mlp_ratio=kv_proj_mlp_ratio,
        )
        self.mlp = ResidualMLP(
            in_dim=channel_dim,
            hidden_dim=int(channel_dim * ffn_mlp_ratio),
            out_dim=channel_dim,
            num_layers=num_layers_ffn,
            act=act,
            input_residual=True,
            output_residual=True,
        )

    def forward(self, x, return_scores: bool = False):
        # x: [B, N, C]
        _x, scores = self.att(self.norm1(x), return_scores=return_scores)
        x = x + _x
        x = x + self.mlp(self.norm2(x))
        return x, scores


class FLAREClassifier(nn.Module):
    """
    FLARE model for multi-label chest X-ray classification
    Uses efficient latent attention mechanism
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
        num_latents: int = 64,
        dropout: float = 0.1,
        pretrained: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.pool = "cls"
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        
        # FLARE transformer blocks
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
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
        
        # Note: FLARE doesn't support pretrained weights for vision yet
        if pretrained:
            print("Warning: FLARE pretrained weights not available. Initializing from scratch.")
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            logits: Classification logits [B, num_classes]
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, embed_dim, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + num_patches, embed_dim]
        x = x + self.pos_embed
        
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(x)
        
        if self.pool == "cls":
            out = x[:, 0]
        elif self.pool == "mean":
            out = x.mean(dim=1)
        else:
            raise ValueError(f"Unknown pool type: {self.pool}")
        
        out = self.head(out)
        return out
