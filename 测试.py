# model_enhanced_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import lpips
from torchvision import transforms
import numpy as np


class EnhancedSemanticEncoder(nn.Module):
    """增强的语义编码器"""

    def __init__(self, model_name='ViT-B-32', local_path='/root/模型2/clip_model/open_clip_model .safetensors'):  # 修改此处
        super().__init__()

        # 修改为
        self.clip_model, self.preprocess = open_clip.create_model_from_pretrained(
            'ViT-B-32',
            pretrained='/root/模型2/clip_model/open_clip_model.safetensors'
        )
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.embed_dim = self.clip_model.visual.output_dim

        self.register_buffer('norm_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('norm_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def forward(self, x, compute_grad=False):
        if compute_grad:
            x_norm = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            x_norm = (x_norm - self.norm_mean) / self.norm_std
            features = self.clip_model.encode_image(x_norm)
            sem_global = F.normalize(features, dim=1)
            B, C = sem_global.shape
            sem_map = sem_global.view(B, C, 1, 1).expand(B, C, 14, 14)
        else:
            with torch.no_grad():
                x_norm = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                x_norm = (x_norm - self.norm_mean) / self.norm_std
                features = self.clip_model.encode_image(x_norm)
                sem_global = F.normalize(features, dim=1)
                B, C = sem_global.shape
                sem_map = sem_global.view(B, C, 1, 1).expand(B, C, 14, 14)

        return sem_global, sem_map


class AdaptiveFiLMLayer(nn.Module):
    """自适应FiLM层，结合了注意力机制"""

    def __init__(self, sem_dim, feature_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(sem_dim, feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim * 2),
        )

        # 自适应尺度控制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 8, feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, sem_global):
        B, C, H, W = x.shape

        # 全局调制
        params = self.mlp(sem_global)
        gamma_global, beta_global = params.chunk(2, dim=1)
        gamma_global = torch.tanh(gamma_global) * 0.2 + 1.0  # [0.8, 1.2]
        beta_global = beta_global * 0.1

        # 局部注意力
        attention_map = self.attention(x)

        # 结合全局和局部调制
        gamma = gamma_global.view(B, C, 1, 1) * attention_map
        beta = beta_global.view(B, C, 1, 1)

        return gamma * x + beta


class MultiScaleAttentionGate(nn.Module):
    """多尺度注意力门控"""

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


class ChannelAttention(nn.Module):
    """通道注意力机制"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class EnhancedResBlock(nn.Module):
    """修复的增强残差块，确保通道匹配"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 通道注意力
        self.ca = ChannelAttention(out_channels)

        # 修复跳跃连接，确保输入输出通道匹配
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)  # 确保identity与输出通道数相同
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out)  # 应用通道注意力
        return self.relu(out + identity)


class EnhancedUNet(nn.Module):
    """修复的增强UNet结构，修正通道数问题"""

    def __init__(self, sem_dim=512, base_channels=32, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        # Encoder - 使用增强的残差块
        self.enc1 = EnhancedResBlock(3, base_channels)
        self.enc2 = EnhancedResBlock(base_channels, base_channels * 2)
        self.enc3 = EnhancedResBlock(base_channels * 2, base_channels * 4)
        self.enc4 = EnhancedResBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            EnhancedResBlock(base_channels * 8, base_channels * 8),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=2, dilation=2),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=4, dilation=4),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )

        # 自适应FiLM注入
        self.film1 = AdaptiveFiLMLayer(sem_dim, base_channels * 4)
        self.film2 = AdaptiveFiLMLayer(sem_dim, base_channels * 2)
        self.film3 = AdaptiveFiLMLayer(sem_dim, base_channels)
        self.film4 = AdaptiveFiLMLayer(sem_dim, base_channels)

        if use_attention:
            self.attention1 = MultiScaleAttentionGate(sem_dim, base_channels * 4)
            self.attention2 = MultiScaleAttentionGate(sem_dim, base_channels * 2)
            self.attention3 = MultiScaleAttentionGate(sem_dim, base_channels)
            self.attention4 = MultiScaleAttentionGate(sem_dim, base_channels)

        # Decoder - 修复通道数问题
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        # 输入通道数: base_channels*4 (上采样) + base_channels*8 (跳跃连接) = 384
        self.dec4 = EnhancedResBlock(base_channels * 4 + base_channels * 8, base_channels * 4)

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        # 输入通道数: base_channels*2 (上采样) + base_channels*4 (跳跃连接) = 192
        self.dec3 = EnhancedResBlock(base_channels * 2 + base_channels * 4, base_channels * 2)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        # 输入通道数: base_channels (上采样) + base_channels*2 (跳跃连接) = 96
        self.dec2 = EnhancedResBlock(base_channels + base_channels * 2, base_channels)

        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        # 输入通道数: base_channels (上采样) + base_channels (跳跃连接) = 64
        self.dec1 = EnhancedResBlock(base_channels + base_channels, base_channels)

        # 输出层 - 简化结构
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Tanh()
        )

        # 自适应残差缩放
        self.residual_scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, sem_global, sem_map):
        # Encoder
        e1 = self.enc1(x)  # [B, 32, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 64, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 128, H/4, W/4]
        e4 = self.enc4(self.pool(e3))  # [B, 256, H/8, W/8]
        b = self.bottleneck(self.pool(e4))  # [B, 256, H/16, W/16]

        # Decoder with adaptive modulation
        d4 = self.up4(b)  # [B, 128, H/8, W/8]
        d4 = self.film1(d4, sem_global)
        if self.use_attention:
            d4 = self.attention1(d4, sem_map)
        # 确保尺寸匹配
        if d4.size() != e4.size():
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)  # [B, 128+256=384, H/8, W/8]
        d4 = self.dec4(d4)  # [B, 128, H/8, W/8]

        d3 = self.up3(d4)  # [B, 64, H/4, W/4]
        d3 = self.film2(d3, sem_global)
        if self.use_attention:
            d3 = self.attention2(d3, sem_map)
        if d3.size() != e3.size():
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)  # [B, 64+128=192, H/4, W/4]
        d3 = self.dec3(d3)  # [B, 64, H/4, W/4]

        d2 = self.up2(d3)  # [B, 32, H/2, W/2]
        d2 = self.film3(d2, sem_global)
        if self.use_attention:
            d2 = self.attention3(d2, sem_map)
        if d2.size() != e2.size():
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)  # [B, 32+64=96, H/2, W/2]
        d2 = self.dec2(d2)  # [B, 32, H/2, W/2]

        d1 = self.up1(d2)  # [B, 32, H, W]
        d1 = self.film4(d1, sem_global)
        if self.use_attention:
            d1 = self.attention4(d1, sem_map)
        if d1.size() != e1.size():
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)  # [B, 32+32=64, H, W]
        d1 = self.dec1(d1)  # [B, 32, H, W]

        # 输出
        residual = self.out_conv(d1) * self.residual_scale
        out = x + residual
        out = torch.clamp(out, 0, 1)

        return out


class EnhancedCognitiveAwareEnhancer(nn.Module):
    """增强的认知感知增强模型"""

    def __init__(self, sem_dim=512, base_channels=32, use_attention=True):
        super().__init__()
        self.semantic_encoder = EnhancedSemanticEncoder()
        self.enhancer = EnhancedUNet(sem_dim=sem_dim, base_channels=base_channels,
                                     use_attention=use_attention)

    def forward(self, x_low, compute_semantic_grad=False):
        sem_global, sem_map = self.semantic_encoder(x_low, compute_grad=compute_semantic_grad)
        y = self.enhancer(x_low, sem_global, sem_map)
        return y


# ==================== 增强的损失函数 ====================

def enhanced_loss_recon(y, gt, alpha=0.85):
    """增强的重建损失"""
    l1_loss = F.l1_loss(y, gt)

    # 使用标准SSIM
    from pytorch_msssim import ssim
    ssim_loss = 1 - ssim(y, gt, data_range=1.0, size_average=True)

    return alpha * l1_loss + (1 - alpha) * ssim_loss


def enhanced_loss_color(y, gt):
    """增强的颜色损失"""
    # 1. 颜色矩匹配
    y_mean = y.mean(dim=[2, 3])
    gt_mean = gt.mean(dim=[2, 3])
    mean_loss = F.l1_loss(y_mean, gt_mean)

    y_std = y.std(dim=[2, 3])
    gt_std = gt.std(dim=[2, 3])
    std_loss = F.l1_loss(y_std, gt_std)

    # 2. 过曝和欠曝惩罚
    over_exposure = F.relu(y - 0.95).mean()
    under_exposure = F.relu(0.05 - y).mean()

    return mean_loss + std_loss + 0.1 * (over_exposure + under_exposure)


def enhanced_loss_smoothness(y):
    """增强的平滑度损失"""
    # 一阶梯度平滑
    dy = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
    dx = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])

    return dy.mean() + dx.mean()


def build_enhanced_model(base_channels=32, use_attention=True):
    """构建增强的模型"""
    model = EnhancedCognitiveAwareEnhancer(sem_dim=512, base_channels=base_channels,
                                           use_attention=use_attention)
    return model