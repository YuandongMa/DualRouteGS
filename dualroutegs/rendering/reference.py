"""Exact finite-ray Gaussian integration for numerical checks and small runs.

O(number of rays * number of Gaussians); not the paper's fast rasterizer.
Chunk checkpointing bounds saved activation memory while keeping gate gradients.
"""
import math
import torch
from torch.utils.checkpoint import checkpoint


def integrate_chunk(means, precision, kappa, origins, directions, lengths):
    offset = origins[:, None] - means[None]
    a = torch.einsum("ri,nij,rj->rn", directions, precision, directions).clamp_min(1e-12)
    b = torch.einsum("ri,nij,rnj->rn", directions, precision, offset)
    c = torch.einsum("rni,nij,rnj->rn", offset, precision, offset)
    perpendicular = (c - b.square() / a).clamp_min(0)
    factor = torch.exp(-.5 * perpendicular) * torch.sqrt(2 * math.pi / a)
    t0 = b / torch.sqrt(2 * a)
    t1 = (a * lengths[:, None] + b) / torch.sqrt(2 * a)
    finite = .5 * (torch.erf(t1) - torch.erf(t0))
    return (factor * finite * kappa[None]).sum(-1)


class ReferenceRenderer:
    def __init__(self, ray_chunk=256, gaussian_chunk=256, checkpoint_chunks=True):
        self.ray_chunk = ray_chunk
        self.gaussian_chunk = gaussian_chunk
        self.checkpoint_chunks = checkpoint_chunks

    def __call__(self, gaussians, cameras, height, width):
        if gaussians.means.shape[0] != len(cameras):
            raise ValueError("One calibrated camera per batch item is required")
        # Radiative integration is evaluated outside AMP, at least in float32.
        dtype = torch.float64 if gaussians.means.dtype == torch.float64 else torch.float32
        with torch.autocast(gaussians.means.device.type, enabled=False):
            g = gaussians.to(dtype=dtype)
            precision = g.precision()
            images = []
            for bi, camera in enumerate(cameras):
                origins, directions, lengths = camera.rays(height, width, g.means.device, dtype)
                pixels = []
                for ri in range(0, height * width, self.ray_chunk):
                    values = torch.zeros_like(lengths[ri:ri+self.ray_chunk])
                    for gi in range(0, g.means.shape[1], self.gaussian_chunk):
                        gs, rs = slice(gi, gi+self.gaussian_chunk), slice(ri, ri+self.ray_chunk)
                        args = (g.means[bi, gs], precision[bi, gs], g.kappa[bi, gs], origins[rs], directions[rs], lengths[rs])
                        if self.checkpoint_chunks and torch.is_grad_enabled() and any(x.requires_grad for x in args[:3]):
                            values = values + checkpoint(integrate_chunk, *args, use_reentrant=False)
                        else:
                            values = values + integrate_chunk(*args)
                    pixels.append(values)
                images.append(torch.cat(pixels).reshape(1, height, width))
            return torch.stack(images)


def voxelize_reference(gaussians, resolution=64, bound=1.0, point_chunk=2048, gaussian_chunk=256):
    """Eq. (41), voxel-center xyz indexing, 3-sigma Mahalanobis truncation."""
    device = gaussians.means.device
    grid = (torch.arange(resolution, device=device, dtype=torch.float32) + .5) * (2*bound/resolution) - bound
    points = torch.stack(torch.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    g = gaussians.to(dtype=torch.float32)
    precision = g.precision()
    volumes = []
    for bi in range(len(g.means)):
        chunks = []
        for start in range(0, len(points), point_chunk):
            pos = points[start:start+point_chunk]
            value = torch.zeros(len(pos), device=device)
            for gi in range(0, g.means.shape[1], gaussian_chunk):
                gs = slice(gi, gi+gaussian_chunk)
                diff = pos[:, None] - g.means[bi, gs]
                dist = torch.einsum("pni,nij,pnj->pn", diff, precision[bi, gs], diff)
                value = value + (torch.exp(-.5*dist) * (dist <= 9) * g.kappa[bi, gs]).sum(-1)
            chunks.append(value)
        volumes.append(torch.cat(chunks).reshape(resolution, resolution, resolution))
    return torch.stack(volumes)
