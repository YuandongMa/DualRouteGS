import torch
import torch.nn.functional as F


def ssim2d(x, y, data_range=1.0, window=11):
    """Gaussian-window SSIM, valid convolution, per-image output [B]."""
    size = min(window, x.shape[-2], x.shape[-1])
    size -= (size+1) % 2
    size = max(size, 1)
    pos = torch.arange(size, device=x.device, dtype=x.dtype) - size//2
    kernel = torch.exp(-pos.square()/(2*1.5**2))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None]*kernel[None, :])[None, None].expand(x.shape[1], 1, size, size)
    filt = lambda value: F.conv2d(value, kernel, groups=x.shape[1])
    mean_x, mean_y = filt(x), filt(y)
    var_x = (filt(x*x)-mean_x.square()).clamp_min(0)
    var_y = (filt(y*y)-mean_y.square()).clamp_min(0)
    cov = filt(x*y)-mean_x*mean_y
    c1, c2 = (.01*data_range)**2, (.03*data_range)**2
    score = (2*mean_x*mean_y+c1)*(2*cov+c2) / ((mean_x.square()+mean_y.square()+c1)*(var_x+var_y+c2))
    return score.flatten(1).mean(1)


def reconstruction_loss(denoised, clean, novel, novel_gt, allocation, cfg, iga_enabled=True):
    den = F.l1_loss(denoised.float(), clean.float())
    nov = F.l1_loss(novel.float(), novel_gt.float())
    structural = 1-ssim2d(novel.float(), novel_gt.float(), cfg["data_range"]).mean()
    if iga_enabled and allocation.probability.shape[-1] > 1:
        extra_ratio = allocation.probability[..., 1:].mean(-1)
        alignment = (extra_ratio-allocation.confidence.detach()).abs().mean()
        budget = (extra_ratio.mean()-cfg["rho_g"]).clamp_min(0)
    else:
        alignment = den*0
        budget = den*0
    total = cfg["den"]*den + cfg["novel"]*nov + cfg["ssim"]*structural + cfg["iga"]*alignment + cfg["budget"]*budget
    return total, {"loss": total, "den_l1": den, "novel_l1": nov, "ssim_loss": structural, "iga_loss": alignment, "budget_loss": budget}
