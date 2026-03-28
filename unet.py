# CA-LLE: semantic-conditioned U-Net decoder
import torch
import torch.nn as nn
import torch.nn.functional as F

from ca_lle.residual import EnhancedResBlock
from ca_lle.conditioning import AdaptiveFiLMLayer, MultiScaleAttentionGate


class EnhancedUNet(nn.Module):
    """U-Net with FiLM and optional semantic attention at each decoder stage."""

    def __init__(self, sem_dim=512, base_channels=32, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        self.enc1 = EnhancedResBlock(3, base_channels)
        self.enc2 = EnhancedResBlock(base_channels, base_channels * 2)
        self.enc3 = EnhancedResBlock(base_channels * 2, base_channels * 4)
        self.enc4 = EnhancedResBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            EnhancedResBlock(base_channels * 8, base_channels * 8),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=2, dilation=2),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=4, dilation=4),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )

        self.film1 = AdaptiveFiLMLayer(sem_dim, base_channels * 4)
        self.film2 = AdaptiveFiLMLayer(sem_dim, base_channels * 2)
        self.film3 = AdaptiveFiLMLayer(sem_dim, base_channels)
        self.film4 = AdaptiveFiLMLayer(sem_dim, base_channels)

        if use_attention:
            self.attention1 = MultiScaleAttentionGate(sem_dim, base_channels * 4)
            self.attention2 = MultiScaleAttentionGate(sem_dim, base_channels * 2)
            self.attention3 = MultiScaleAttentionGate(sem_dim, base_channels)
            self.attention4 = MultiScaleAttentionGate(sem_dim, base_channels)

        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec4 = EnhancedResBlock(base_channels * 4 + base_channels * 8, base_channels * 4)

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec3 = EnhancedResBlock(base_channels * 2 + base_channels * 4, base_channels * 2)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec2 = EnhancedResBlock(base_channels + base_channels * 2, base_channels)

        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.dec1 = EnhancedResBlock(base_channels + base_channels, base_channels)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Tanh(),
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, sem_global, sem_map):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.film1(d4, sem_global)
        if self.use_attention:
            d4 = self.attention1(d4, sem_map)
        if d4.size() != e4.size():
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self.film2(d3, sem_global)
        if self.use_attention:
            d3 = self.attention2(d3, sem_map)
        if d3.size() != e3.size():
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.film3(d2, sem_global)
        if self.use_attention:
            d2 = self.attention3(d2, sem_map)
        if d2.size() != e2.size():
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.film4(d1, sem_global)
        if self.use_attention:
            d1 = self.attention4(d1, sem_map)
        if d1.size() != e1.size():
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        residual = self.out_conv(d1) * self.residual_scale
        out = x + residual
        out = torch.clamp(out, 0, 1)

        return out
