# main_enhanced_fixed.py
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import cv2
import lpips
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from model_enhanced import (
    build_enhanced_model, enhanced_loss_recon, enhanced_loss_color,
    enhanced_loss_smoothness
)


class EnhancedLowLightDataset(Dataset):
    """增强的低光数据集"""

    def __init__(self, low_dir, gt_dir=None, size=256, is_train=True):
        self.low_dir = Path(low_dir)
        self.gt_dir = Path(gt_dir) if gt_dir else None
        self.size = size
        self.is_train = is_train

        self.low_files = sorted(list(self.low_dir.glob('*.png')) +
                                list(self.low_dir.glob('*.jpg')))

        if self.gt_dir:
            self.paired = []
            gt_files = sorted(list(self.gt_dir.glob('*.png')) +
                              list(self.gt_dir.glob('*.jpg')))
            gt_dict = {f.stem: f for f in gt_files}

            for low_file in self.low_files:
                stem = low_file.stem
                if stem in gt_dict:
                    self.paired.append((low_file, gt_dict[stem]))
                else:
                    print(f"Warning: GT not found for {low_file.name}")
            print(f"Found {len(self.paired)} paired images")
        else:
            self.paired = None
            print(f"Found {len(self.low_files)} unpaired images")

    def __len__(self):
        return len(self.paired) if self.paired else len(self.low_files)

    def __getitem__(self, idx):
        if self.paired:
            low_path, gt_path = self.paired[idx]
            low_img = Image.open(low_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB')
        else:
            low_path = self.low_files[idx]
            low_img = Image.open(low_path).convert('RGB')
            gt_img = None

        if self.is_train:
            if gt_img:
                # 随机裁剪
                i, j, h, w = transforms.RandomCrop.get_params(
                    low_img, output_size=(self.size, self.size)
                )
                low_img = transforms.functional.crop(low_img, i, j, h, w)
                gt_img = transforms.functional.crop(gt_img, i, j, h, w)

                # 随机翻转
                if random.random() > 0.5:
                    low_img = transforms.functional.hflip(low_img)
                    gt_img = transforms.functional.hflip(gt_img)
                if random.random() > 0.5:
                    low_img = transforms.functional.vflip(low_img)
                    gt_img = transforms.functional.vflip(gt_img)
            else:
                low_img = transforms.RandomCrop(self.size)(low_img)
                if random.random() > 0.5:
                    low_img = transforms.functional.hflip(low_img)
                if random.random() > 0.5:
                    low_img = transforms.functional.vflip(low_img)
        else:
            low_img = transforms.Resize((self.size, self.size))(low_img)
            if gt_img:
                gt_img = transforms.Resize((self.size, self.size))(gt_img)

        low_tensor = transforms.ToTensor()(low_img)
        gt_tensor = transforms.ToTensor()(gt_img) if gt_img else None

        if gt_tensor is not None:
            return low_tensor, gt_tensor, str(low_path.name)
        else:
            return low_tensor, str(low_path.name)


class AdaptiveEarlyStopping:
    """自适应早停机制"""

    def __init__(self, patience=20, min_delta=0.002, warmup=10):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric, epoch):
        if epoch < self.warmup:
            return False

        score = val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


def train_enhanced_epoch(model, loader, optimizer, device, args, lpips_model, epoch):
    """修复的训练epoch"""
    model.train()
    model.semantic_encoder.eval()

    total_loss = 0
    loss_components = {'rec': 0, 'col': 0, 'smooth': 0, 'lpips': 0}
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

        # 损失计算
        loss, loss_dict = compute_enhanced_loss(y, gt, lpips_model, args, epoch)

        if loss.item() == 0 or torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_components[k] += v

        # 动态更新进度条
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'rec': f"{loss_dict.get('rec', 0):.3f}",
                'col': f"{loss_dict.get('col', 0):.3f}",
            })

    avg_loss = total_loss / len(loader)
    avg_components = {k: v / len(loader) for k, v in loss_components.items()}

    return avg_loss, avg_components


def compute_enhanced_loss(y, gt, lpips_model, args, epoch):
    """修复的损失计算"""
    loss = 0
    loss_dict = {}

    if y.isnan().any() or y.isinf().any():
        return torch.tensor(0.0, device=y.device, requires_grad=True), {}

    if gt is not None:
        # 动态损失权重调度
        epoch_ratio = min(epoch / args.epochs, 1.0)

        # 重建损失
        l_rec = enhanced_loss_recon(y, gt)
        rec_weight = 1.0
        loss += rec_weight * l_rec
        loss_dict['rec'] = l_rec.item()

        # 颜色损失
        l_col = enhanced_loss_color(y, gt)
        col_weight = 0.8
        loss += col_weight * l_col
        loss_dict['col'] = l_col.item()

        # 平滑度损失
        l_smooth = enhanced_loss_smoothness(y)
        smooth_weight = 0.05
        loss += smooth_weight * l_smooth
        loss_dict['smooth'] = l_smooth.item()

        # LPIPS损失
        if args.use_lpips and lpips_model is not None:
            l_lp = lpips_model(y * 2 - 1, gt * 2 - 1).mean()
            lpips_weight = 0.05
            loss += lpips_weight * l_lp
            loss_dict['lpips'] = l_lp.item()

    if loss.isnan() or loss.isinf():
        return torch.tensor(0.0, device=y.device, requires_grad=True), {}

    return loss, loss_dict


def validate_enhanced(model, loader, device, lpips_model):
    """修复的验证函数"""
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

            # 计算指标
            for i in range(y.shape[0]):
                y_np = y[i].cpu().numpy().transpose(1, 2, 0)
                gt_np = gt[i].cpu().numpy().transpose(1, 2, 0)

                # PSNR
                mse = np.mean((y_np - gt_np) ** 2)
                if mse > 0:
                    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
                    psnr_list.append(psnr)

                # SSIM
                from skimage.metrics import structural_similarity as ssim_func
                ssim_val = ssim_func(y_np, gt_np, data_range=1.0, channel_axis=2)
                ssim_list.append(ssim_val)

            # LPIPS
            if lpips_model:
                l_lp = lpips_model(y * 2 - 1, gt * 2 - 1).mean()
                lpips_list.append(l_lp.item())

    metrics = {
        'psnr': np.mean(psnr_list) if psnr_list else 0,
        'ssim': np.mean(ssim_list) if ssim_list else 0,
        'lpips': np.mean(lpips_list) if lpips_list else 0
    }

    return metrics


def train_enhanced(args):
    """修复的训练主函数"""
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # 构建增强的模型
    model = build_enhanced_model(base_channels=32, use_attention=True).to(device)
    print(f"Enhanced model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 损失模型
    lpips_model = lpips.LPIPS(net='alex').to(device) if args.use_lpips else None

    # 数据加载器
    train_dataset = EnhancedLowLightDataset(
        args.train_low,
        args.train_gt if not args.self_supervised else None,
        size=256,
        is_train=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = None
    if args.val_low and args.val_gt:
        val_dataset = EnhancedLowLightDataset(args.val_low, args.val_gt, size=256, is_train=False)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.02
    )

    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 早停机制
    early_stopping = AdaptiveEarlyStopping(patience=20, min_delta=0.001, warmup=10)

    best_ssim = 0
    log_file = open(os.path.join(args.out_dir, 'log_enhanced.txt'), 'w')

    print("Starting enhanced training...")

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, loss_components = train_enhanced_epoch(
            model, train_loader, optimizer, device, args, lpips_model, epoch
        )

        current_lr = optimizer.param_groups[0]['lr']
        log_msg = f"Epoch {epoch}/{args.epochs} - LR: {current_lr:.2e} - Train Loss: {train_loss:.4f}"

        comp_str = " | ".join([f"{k}: {v:.4f}" for k, v in loss_components.items() if v > 0])
        log_msg += f" ({comp_str})"

        # 验证
        if val_loader and epoch % args.val_interval == 0:
            metrics = validate_enhanced(model, val_loader, device, lpips_model)
            log_msg += f" | Val PSNR: {metrics['psnr']:.2f} SSIM: {metrics['ssim']:.4f}"
            if args.use_lpips:
                log_msg += f" LPIPS: {metrics['lpips']:.4f}"

            # 保存最佳模型
            if metrics['ssim'] > best_ssim:
                best_ssim = metrics['ssim']
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'ssim': best_ssim,
                    'psnr': metrics['psnr'],
                    'args': vars(args)
                }, os.path.join(args.out_dir, 'best_enhanced.ckpt'))
                log_msg += " [BEST]"

            # 早停检查
            if early_stopping(metrics['ssim'], epoch):
                print("Early stopping triggered!")
                log_file.write("Early stopping triggered!\n")
                break

        print(log_msg)
        log_file.write(log_msg + '\n')
        log_file.flush()

        # 定期保存检查点
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.out_dir, f'ckpt_epoch{epoch}_enhanced.ckpt'))

        scheduler.step()

    log_file.close()
    print(f"Enhanced training completed! Best SSIM: {best_ssim:.4f}")


def infer_enhanced(args):
    """修复的推理函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_enhanced_model(base_channels=32, use_attention=True).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f"Loaded enhanced checkpoint from {args.ckpt}")
    if 'ssim' in ckpt:
        print(f"Model SSIM: {ckpt['ssim']:.4f}, PSNR: {ckpt.get('psnr', 0):.2f}")

    os.makedirs(args.save_dir, exist_ok=True)

    input_dir = Path(args.input_dir)
    image_files = sorted(list(input_dir.glob('*.png')) +
                         list(input_dir.glob('*.jpg')) +
                         list(input_dir.glob('*.jpeg')))

    print(f"Processing {len(image_files)} images with enhanced model...")

    with torch.no_grad():
        for img_path in tqdm(image_files):
            img = Image.open(img_path).convert('RGB')
            original_size = img.size
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

            enhanced = model(img_tensor, compute_semantic_grad=False)

            enhanced_np = enhanced[0].cpu().numpy().transpose(1, 2, 0)
            enhanced_np = np.clip(enhanced_np, 0, 1)

            # 高质量保存
            enhanced_uint8 = (enhanced_np * 255).astype(np.uint8)
            enhanced_img = Image.fromarray(enhanced_uint8)

            if enhanced_img.size != original_size:
                enhanced_img = enhanced_img.resize(original_size, Image.LANCZOS)

            save_path = os.path.join(args.save_dir, img_path.name)
            enhanced_img.save(save_path, quality=95)

    print(f"Enhanced inference completed! Results saved to {args.save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Low-Light Enhancement')

    # 模式
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'])

    # 数据路径
    parser.add_argument('--train_low', type=str, default=None)
    parser.add_argument('--train_gt', type=str, default=None)
    parser.add_argument('--val_low', type=str, default=None)
    parser.add_argument('--val_gt', type=str, default=None)
    parser.add_argument('--input_dir', type=str, default=None)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # 模型配置
    parser.add_argument('--use_lpips', type=int, default=1)
    parser.add_argument('--self_supervised', type=int, default=0)

    # 输出
    parser.add_argument('--out_dir', type=str, default='runs/enhanced_exp')
    parser.add_argument('--save_dir', type=str, default='outputs/enhanced')
    parser.add_argument('--val_interval', type=int, default=5)

    # 推理
    parser.add_argument('--ckpt', type=str, default=None)

    args = parser.parse_args()

    if args.mode == 'train':
        if not args.train_low:
            raise ValueError("--train_low is required for training")
        train_enhanced(args)
    elif args.mode == 'infer':
        if not args.ckpt or not args.input_dir:
            raise ValueError("--ckpt and --input_dir are required for inference")
        infer_enhanced(args)


if __name__ == '__main__':
    main()