# CA-LLE: full cognitive-aware enhancer (CLIP + U-Net)
import torch.nn as nn

from ca_lle.semantic_encoder import EnhancedSemanticEncoder
from ca_lle.unet import EnhancedUNet


class EnhancedCognitiveAwareEnhancer(nn.Module):
    """CLIP semantics + residual U-Net for low-light enhancement."""

    def __init__(self, sem_dim=512, base_channels=32, use_attention=True):
        super().__init__()
        self.semantic_encoder = EnhancedSemanticEncoder()
        self.enhancer = EnhancedUNet(
            sem_dim=sem_dim,
            base_channels=base_channels,
            use_attention=use_attention,
        )

    def forward(self, x_low, compute_semantic_grad=False):
        sem_global, sem_map = self.semantic_encoder(
            x_low, compute_grad=compute_semantic_grad
        )
        y = self.enhancer(x_low, sem_global, sem_map)
        return y


def build_enhanced_model(base_channels=32, use_attention=True):
    """Build enhancer with default LAION ViT-B-32 semantics (512-D)."""
    return EnhancedCognitiveAwareEnhancer(
        sem_dim=512,
        base_channels=base_channels,
        use_attention=use_attention,
    )
