# CA-LLE: full training loop
import os

import lpips
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ca_lle.callbacks import AdaptiveEarlyStopping
from ca_lle.dataset import EnhancedLowLightDataset
from ca_lle.enhancer import build_enhanced_model
from ca_lle.train_utils import train_enhanced_epoch, validate_enhanced


def train_enhanced(args):
    """Training with optional paired validation and cosine LR."""
    import random
    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    model = build_enhanced_model(base_channels=32, use_attention=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Enhanced model parameters: {n_params:,}")

    lpips_model = lpips.LPIPS(net='alex').to(device) if args.use_lpips else None

    train_dataset = EnhancedLowLightDataset(
        args.train_low,
        args.train_gt if not args.self_supervised else None,
        size=256,
        is_train=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = None
    if args.val_low and args.val_gt:
        val_dataset = EnhancedLowLightDataset(
            args.val_low, args.val_gt, size=256, is_train=False
        )
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.02,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    early_stopping = AdaptiveEarlyStopping(patience=20, min_delta=0.001, warmup=10)

    best_ssim = 0
    log_file = open(os.path.join(args.out_dir, 'log_enhanced.txt'), 'w')

    print("Starting enhanced training...")

    for epoch in range(1, args.epochs + 1):
        train_loss, loss_components = train_enhanced_epoch(
            model, train_loader, optimizer, device, args, lpips_model, epoch
        )

        current_lr = optimizer.param_groups[0]['lr']
        log_msg = (
            f"Epoch {epoch}/{args.epochs} - LR: {current_lr:.2e} - "
            f"Train Loss: {train_loss:.4f}"
        )

        comp_str = " | ".join(
            f"{k}: {v:.4f}" for k, v in loss_components.items() if v > 0
        )
        log_msg += f" ({comp_str})"

        if val_loader and epoch % args.val_interval == 0:
            metrics = validate_enhanced(model, val_loader, device, lpips_model)
            log_msg += (
                f" | Val PSNR: {metrics['psnr']:.2f} SSIM: {metrics['ssim']:.4f}"
            )
            if args.use_lpips:
                log_msg += f" LPIPS: {metrics['lpips']:.4f}"

            if metrics['ssim'] > best_ssim:
                best_ssim = metrics['ssim']
                torch.save(
                    {
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'ssim': best_ssim,
                        'psnr': metrics['psnr'],
                        'args': vars(args),
                    },
                    os.path.join(args.out_dir, 'best_enhanced.ckpt'),
                )
                log_msg += " [BEST]"

            if early_stopping(metrics['ssim'], epoch):
                print("Early stopping triggered!")
                log_file.write("Early stopping triggered!\n")
                break

        print(log_msg)
        log_file.write(log_msg + '\n')
        log_file.flush()

        if epoch % 20 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                os.path.join(args.out_dir, f'ckpt_epoch{epoch}_enhanced.ckpt'),
            )

        scheduler.step()

    log_file.close()
    print(f"Enhanced training completed! Best SSIM: {best_ssim:.4f}")
