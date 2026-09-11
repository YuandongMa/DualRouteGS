"""Metrics are computed from predictions; no manuscript scores are pre-filled."""
import math
import numpy as np
import torch
from .losses import ssim2d


def projection_metrics(prediction, truth, data_range=1.0):
    error = (prediction.float()-truth.float()).square().flatten(1).mean(1)
    psnr = 10*torch.log10(data_range**2/error.clamp_min(1e-12))
    ssim = ssim2d(prediction.float(), truth.float(), data_range)
    return {"psnr": float(psnr.mean()), "ssim": float(ssim.mean()), "rmse": float(error.sqrt().mean())}


def volume_metrics(prediction, truth, data_range=1.0):
    from scipy.ndimage import gaussian_filter
    x, y = np.asarray(prediction, np.float64), np.asarray(truth, np.float64)
    if x.shape != y.shape or x.ndim != 3:
        raise ValueError("Volume metrics require matching 3D grids")
    mx, my = gaussian_filter(x, 1.5), gaussian_filter(y, 1.5)
    vx = np.maximum(gaussian_filter(x*x, 1.5)-mx*mx, 0)
    vy = np.maximum(gaussian_filter(y*y, 1.5)-my*my, 0)
    covariance = gaussian_filter(x*y, 1.5)-mx*my
    c1, c2 = (.01*data_range)**2, (.03*data_range)**2
    ssim = ((2*mx*my+c1)*(2*covariance+c2)/((mx*mx+my*my+c1)*(vx+vy+c2))).mean()
    mse = ((x-y)**2).mean()
    return {"volume_psnr": float(10*math.log10(data_range**2/max(mse, 1e-12))), "volume_ssim3d": float(ssim), "volume_rmse": float(np.sqrt(mse))}


def bootstrap_mean(values, seed=0, resamples=10000):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty split")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(resamples, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(), "n": len(values)}
