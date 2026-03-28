# CA-LLE: frozen CLIP image encoder for scene semantics
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class EnhancedSemanticEncoder(nn.Module):
    """Frozen CLIP ViT image encoder; returns global vector and tiled ``sem_map``."""

    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'):
        super().__init__()
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.embed_dim = self.clip_model.visual.output_dim

        self.register_buffer(
            'norm_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            'norm_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

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
