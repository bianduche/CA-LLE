# CA-LLE: batch training / validation utilities
import numpy as np
import torch
from tqdm import tqdm

from ca_lle.losses import (
    enhanced_loss_color,
    enhanced_loss_recon,
    enhanced_loss_self_supervised,
    enhanced_loss_smoothness,
)


def train_enhanced_epoch(model, loader, optimizer, device, args, lpips_model, epoch):
    """One training epoch with mixed reconstruction / perceptual / self-supervised losses."""
    model.train()
    model.semantic_encoder.eval()

    total_loss = 0
    loss_components = {'rec': 0, 'col': 0, 'smooth': 0, 'lpips': 0, 'self': 0}
    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        if args.self_supervised or not args.train_gt:
            x_low, _ = batch
            x_low = x_low.to(device)
            gt = None
        else:
            x_low, gt, _ = batch
            x_low = x_low.to(device)
            gt = gt.to(device)

        optimizer.zero_grad()

        y = model(x_low, compute_semantic_grad=False)

        loss, loss_dict = compute_enhanced_loss(
            y, gt, lpips_model, args, epoch, x_low=x_low
        )

        if loss.item() == 0 or torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0) + v

        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'rec': f"{loss_dict.get('rec', 0):.3f}",
                'col': f"{loss_dict.get('col', 0):.3f}",
            })

    avg_loss = total_loss / len(loader)
    avg_components = {k: v / len(loader) for k, v in loss_components.items()}

    return avg_loss, avg_components


def compute_enhanced_loss(y, gt, lpips_model, args, epoch, x_low=None):
    """Aggregate supervised terms or self-supervision when ``gt`` is missing."""
    del epoch  # reserved for curriculum / scheduling
    loss = None
    loss_dict = {}

    if y.isnan().any() or y.isinf().any():
        return torch.tensor(0.0, device=y.device, requires_grad=True), {}

    if gt is not None:
        loss = 0
        l_rec = enhanced_loss_recon(y, gt)
        rec_weight = 1.0
        loss += rec_weight * l_rec
        loss_dict['rec'] = l_rec.item()

        l_col = enhanced_loss_color(y, gt)
        col_weight = 0.8
        loss += col_weight * l_col
        loss_dict['col'] = l_col.item()

        l_smooth = enhanced_loss_smoothness(y)
        smooth_weight = 0.05
        loss += smooth_weight * l_smooth
        loss_dict['smooth'] = l_smooth.item()

        if args.use_lpips and lpips_model is not None:
            l_lp = lpips_model(y * 2 - 1, gt * 2 - 1).mean()
            lpips_weight = 0.05
            loss += lpips_weight * l_lp
            loss_dict['lpips'] = l_lp.item()

    elif x_low is not None:
        l_self = enhanced_loss_self_supervised(y, x_low)
        loss = l_self
        loss_dict['self'] = l_self.item()
        if args.use_lpips and lpips_model is not None:
            l_lp = lpips_model(y * 2 - 1, x_low * 2 - 1).mean()
            lpips_weight = 0.02
            loss = loss + lpips_weight * l_lp
            loss_dict['lpips'] = l_lp.item()

    if loss is None:
        return torch.tensor(0.0, device=y.device, requires_grad=True), {}

    if loss.isnan() or loss.isinf():
        return torch.tensor(0.0, device=y.device, requires_grad=True), {}

    return loss, loss_dict


def validate_enhanced(model, loader, device, lpips_model):
    """Validation: PSNR, SSIM, optional LPIPS (requires paired loader)."""
    model.eval()

    psnr_list = []
    ssim_list = []
    lpips_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            x_low, gt, _ = batch
            x_low = x_low.to(device)
            gt = gt.to(device)

            y = model(x_low, compute_semantic_grad=False)

            for i in range(y.shape[0]):
                y_np = y[i].cpu().numpy().transpose(1, 2, 0)
                gt_np = gt[i].cpu().numpy().transpose(1, 2, 0)

                mse = np.mean((y_np - gt_np) ** 2)
                if mse > 0:
                    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
                    psnr_list.append(psnr)

                from skimage.metrics import structural_similarity as ssim_func

                ssim_val = ssim_func(y_np, gt_np, data_range=1.0, channel_axis=2)
                ssim_list.append(ssim_val)

            if lpips_model:
                l_lp = lpips_model(y * 2 - 1, gt * 2 - 1).mean()
                lpips_list.append(l_lp.item())

    return {
        'psnr': np.mean(psnr_list) if psnr_list else 0,
        'ssim': np.mean(ssim_list) if ssim_list else 0,
        'lpips': np.mean(lpips_list) if lpips_list else 0,
    }
