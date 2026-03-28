# CA-LLE: Cognitive-Aware Low-Light Enhancement
"""Modular reference implementation: semantic prior + U-Net enhancement."""

from ca_lle.semantic_encoder import EnhancedSemanticEncoder
from ca_lle.conditioning import AdaptiveFiLMLayer, MultiScaleAttentionGate
from ca_lle.residual import ChannelAttention, EnhancedResBlock
from ca_lle.unet import EnhancedUNet
from ca_lle.enhancer import EnhancedCognitiveAwareEnhancer, build_enhanced_model
from ca_lle.losses import (
    enhanced_loss_recon,
    enhanced_loss_color,
    enhanced_loss_smoothness,
    enhanced_loss_self_supervised,
)

__all__ = [
    'EnhancedSemanticEncoder',
    'AdaptiveFiLMLayer',
    'MultiScaleAttentionGate',
    'ChannelAttention',
    'EnhancedResBlock',
    'EnhancedUNet',
    'EnhancedCognitiveAwareEnhancer',
    'build_enhanced_model',
    'enhanced_loss_recon',
    'enhanced_loss_color',
    'enhanced_loss_smoothness',
    'enhanced_loss_self_supervised',
]

__version__ = '1.0.0'
