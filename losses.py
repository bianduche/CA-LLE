# CA-LLE: supervised and self-supervised losses
import torch
import torch.nn.functional as F


def enhanced_loss_recon(y, gt, alpha=0.85):
    """L1 + (1 - SSIM) reconstruction loss."""
    l1_loss = F.l1_loss(y, gt)

    from pytorch_msssim import ssim

    ssim_loss = 1 - ssim(y, gt, data_range=1.0, size_average=True)

    return alpha * l1_loss + (1 - alpha) * ssim_loss


def enhanced_loss_color(y, gt):
    """Channel moments + mild exposure regularization."""
    y_mean = y.mean(dim=[2, 3])
    gt_mean = gt.mean(dim=[2, 3])
    mean_loss = F.l1_loss(y_mean, gt_mean)

    y_std = y.std(dim=[2, 3])
    gt_std = gt.std(dim=[2, 3])
    std_loss = F.l1_loss(y_std, gt_std)

    over_exposure = F.relu(y - 0.95).mean()
    under_exposure = F.relu(0.05 - y).mean()

    return mean_loss + std_loss + 0.1 * (over_exposure + under_exposure)


def enhanced_loss_smoothness(y):
    """Total variation–style smoothness on the output."""
    dy = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
    dx = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])

    return dy.mean() + dx.mean()


def enhanced_loss_self_supervised(y, x_low):
    """Unpaired: structure vs. input, smoothness, no darker than input."""
    from pytorch_msssim import ssim

    l_ssim = 1 - ssim(y, x_low, data_range=1.0, size_average=True)
    l_smooth = enhanced_loss_smoothness(y)
    no_darker = F.relu(x_low - y).mean()

    y_l = 0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3]
    x_l = 0.299 * x_low[:, 0:1] + 0.587 * x_low[:, 1:2] + 0.114 * x_low[:, 2:3]
    lift = F.relu(x_l - y_l).mean()

    return 0.55 * l_ssim + 0.15 * l_smooth + 0.2 * no_darker + 0.1 * lift
