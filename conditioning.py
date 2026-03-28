# CA-LLE: FiLM and semantic attention gates
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFiLMLayer(nn.Module):
    """FiLM modulation from global semantics with a spatial attention map."""

    def __init__(self, sem_dim, feature_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(sem_dim, feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim * 2),
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 8, feature_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, sem_global):
        B, C, H, W = x.shape

        params = self.mlp(sem_global)
        gamma_global, beta_global = params.chunk(2, dim=1)
        gamma_global = torch.tanh(gamma_global) * 0.2 + 1.0
        beta_global = beta_global * 0.1

        attention_map = self.attention(x)

        gamma = gamma_global.view(B, C, 1, 1) * attention_map
        beta = beta_global.view(B, C, 1, 1)

        return gamma * x + beta


class MultiScaleAttentionGate(nn.Module):
    """Gates feature maps using upsampled semantic features."""

    def __init__(self, sem_dim, feature_dim):
        super().__init__()
        self.proj1 = nn.Conv2d(sem_dim, feature_dim // 4, 1)
        self.proj2 = nn.Conv2d(feature_dim // 4, feature_dim, 1)
        self.bn = nn.BatchNorm2d(feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, sem_map):
        sem_up = F.interpolate(sem_map, size=x.shape[2:], mode='bilinear', align_corners=False)
        gate = self.proj2(F.relu(self.proj1(sem_up)))
        gate = self.sigmoid(self.bn(gate))
        return x * gate
