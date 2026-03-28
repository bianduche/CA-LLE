# CA-LLE: batched inference on a folder of images
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from ca_lle.enhancer import build_enhanced_model


def infer_enhanced(args):
    """Run the enhancer on all images in ``input_dir``; save to ``save_dir``."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_enhanced_model(base_channels=32, use_attention=True).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f"Loaded enhanced checkpoint from {args.ckpt}")
    if 'ssim' in ckpt:
        print(
            f"Model SSIM: {ckpt['ssim']:.4f}, PSNR: {ckpt.get('psnr', 0):.2f}"
        )

    os.makedirs(args.save_dir, exist_ok=True)

    input_dir = Path(args.input_dir)
    image_files = sorted(
        list(input_dir.glob('*.png'))
        + list(input_dir.glob('*.jpg'))
        + list(input_dir.glob('*.jpeg'))
    )

    print(f"Processing {len(image_files)} images with enhanced model...")

    with torch.no_grad():
        for img_path in tqdm(image_files):
            img = Image.open(img_path).convert('RGB')
            original_size = img.size
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

            enhanced = model(img_tensor, compute_semantic_grad=False)

            enhanced_np = enhanced[0].cpu().numpy().transpose(1, 2, 0)
            enhanced_np = np.clip(enhanced_np, 0, 1)

            enhanced_uint8 = (enhanced_np * 255).astype(np.uint8)
            enhanced_img = Image.fromarray(enhanced_uint8)

            if enhanced_img.size != original_size:
                enhanced_img = enhanced_img.resize(original_size, Image.LANCZOS)

            save_path = os.path.join(args.save_dir, img_path.name)
            enhanced_img.save(save_path, quality=95)

    print(f"Enhanced inference completed! Results saved to {args.save_dir}")
