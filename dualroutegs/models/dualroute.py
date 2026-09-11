"""Camera-conditioned denoiser predicting one global attenuation Gaussian field."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from ..geometry import ray_features
from .psr import PhaseAwareStateRouting, scalar_embedding
from .iga import InnovationGuidedAllocator


def rope_2d(tokens, height, width):
    dim = tokens.shape[-1]
    row, col = torch.meshgrid(torch.arange(height, device=tokens.device), torch.arange(width, device=tokens.device), indexing="ij")
    coord = torch.stack([row.flatten(), col.flatten()], -1).float()
    freq = 10000**(-torch.arange(dim//4, device=tokens.device).float() / max(dim//4, 1))
    angle = (coord[..., None]*freq).reshape(height*width, dim//2)
    pair = tokens.float().reshape(*tokens.shape[:-1], dim//2, 2)
    real = pair[..., 0]*angle.cos() - pair[..., 1]*angle.sin()
    imag = pair[..., 0]*angle.sin() + pair[..., 1]*angle.cos()
    return torch.stack([real, imag], -1).flatten(-2).to(tokens.dtype)


class DualRouteBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg["dim"]
        self.interaction = cfg["interaction"]
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6*dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        with torch.no_grad():
            self.modulation[-1].bias[2*dim:3*dim] = .1
            self.modulation[-1].bias[5*dim:] = .1
        if self.interaction == "attention":
            self.operation = nn.MultiheadAttention(dim, cfg["heads"], batch_first=True)
        else:
            self.operation = PhaseAwareStateRouting(cfg)
        hidden = int(dim*cfg["mlp_ratio"])
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))

    def forward(self, x, condition, log_snr, memory):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).chunk(6, -1)
        normalized = self.norm1(x)*(1+scale1[:, None]) + shift1[:, None]
        if self.interaction == "attention":
            output = self.operation(normalized, normalized, normalized, need_weights=False)[0]
            diag = {"confidence": x.new_full(x.shape[:2], .5), "query_fraction": x.new_ones(x.shape[0]), "active_subspaces": x.new_zeros(x.shape[:2]), "reliability": x.new_ones(x.shape[0])}
        else:
            output, memory, diag = self.operation(normalized, log_snr, memory)
        y = x + gate1[:, None]*output
        z = self.norm2(y)*(1+scale2[:, None]) + shift2[:, None]
        return y+gate2[:, None]*self.ffn(z), memory, diag


class DualRouteGS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dim, patch = cfg["dim"], cfg["patch_size"]
        self.target_encoder = nn.Conv2d(7, dim, patch, stride=patch)
        self.condition_encoder = nn.Conv2d(7, dim, patch, stride=patch)
        self.stream_embedding = nn.Parameter(torch.randn(2, dim)*.02)
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([DualRouteBlock(cfg) for _ in range(cfg["depth"])])
        self.norm = nn.LayerNorm(dim)
        self.allocator = InnovationGuidedAllocator(cfg)

    def forward(self, noisy_target, measured, target_cameras, condition_cameras, log_snr):
        if noisy_target.ndim != 4 or measured.ndim != 4 or noisy_target.shape != measured.shape or measured.shape[1] != 1:
            raise ValueError("Require noisy_target and exactly one measured projection per item, both [B,1,H,W]")
        b, _, height, width = measured.shape
        if height % self.cfg["patch_size"] or width % self.cfg["patch_size"]:
            raise ValueError("Projection size must be divisible by patch size")
        patch = self.cfg["patch_size"]
        gh, gw = height//patch, width//patch
        targets = ray_features(target_cameras, height, width, measured.device, measured.dtype)
        conditions = ray_features(condition_cameras, height, width, measured.device, measured.dtype)
        xt = self.target_encoder(torch.cat([noisy_target, targets], 1)).flatten(2).transpose(1, 2)
        xc = self.condition_encoder(torch.cat([measured, conditions], 1)).flatten(2).transpose(1, 2)
        x = torch.cat([rope_2d(xt+self.stream_embedding[0], gh, gw), rope_2d(xc+self.stream_embedding[1], gh, gw)], 1)
        condition = self.time(scalar_embedding(log_snr, self.cfg["dim"]))
        # Memory resets for every sample/denoising call; only depth-wise sharing.
        memory = None
        diagnostics = []
        for block in self.blocks:
            if self.training and self.cfg.get("gradient_checkpointing", False):
                x, memory, diag = checkpoint(block, x, condition, log_snr, memory, use_reentrant=False)
            else:
                x, memory, diag = block(x, condition, log_snr, memory)
            diagnostics.append(diag)
        target = self.norm(x[:, :gh*gw]).transpose(1, 2).reshape(b, self.cfg["dim"], gh, gw)
        confidence = diagnostics[-1]["confidence"][:, :gh*gw].reshape(b, gh, gw)
        return self.allocator(target, confidence, target_cameras), diagnostics
