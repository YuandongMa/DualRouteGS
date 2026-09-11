"""Innovation-Guided Gaussian Allocation, manuscript Eqs. (28)-(35)."""
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from ..geometry import ray_box_interval
from ..gaussians import GaussianSet
from .psr import straight_through


@dataclass
class Allocation:
    gaussians: GaussianSet
    probability: torch.Tensor  # [B,anchors,K], unthresholded soft gates
    hard: torch.Tensor
    st: torch.Tensor
    confidence: torch.Tensor

    def relaxed(self):
        return self.gaussians.weighted(self.st.flatten(1))

    def active(self):
        return [self.gaussians.select(b, self.hard[b].flatten()) for b in range(self.hard.shape[0])]


class InnovationGuidedAllocator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.size, self.k_max = cfg["anchor_size"], cfg["k_max"]
        width = cfg["decoder_dim"]
        self.reduce = nn.Sequential(nn.Conv2d(cfg["dim"], width, 1), nn.SiLU())
        self.refine = nn.Sequential(nn.Conv2d(width, width, 3, padding=1, groups=width), nn.SiLU(), nn.Conv2d(width, width, 1))
        self.parameters_head = nn.Conv2d(width, 11*self.k_max, 1)
        self.score = nn.Conv2d(width, 1, 1)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.zeta2 = nn.Parameter(torch.tensor(0.0))
        self.raw_gaps = nn.Parameter(torch.zeros(max(self.k_max-2, 0)))
        nn.init.normal_(self.parameters_head.weight, std=.001)
        nn.init.zeros_(self.parameters_head.bias)
        with torch.no_grad():
            bias = self.parameters_head.bias.reshape(self.k_max, 11)
            bias[:, 3] = 1.0  # quaternion w
            bias[:, 7:10] = -2.5  # moderate initial scale
            bias[:, 10] = math.log(math.expm1(cfg["initial_kappa"]))

    def ordered_thresholds(self):
        if self.k_max <= 1:
            return self.raw_gaps.new_empty(0)
        return torch.cat([self.zeta2[None], self.zeta2 + F.softplus(self.raw_gaps).cumsum(0)])

    def forward(self, features, confidence, target_cameras):
        b = features.shape[0]
        z = F.interpolate(self.reduce(features), (self.size, self.size), mode="bilinear", align_corners=False)
        z = self.refine(z)
        p = F.interpolate(confidence[:, None], (self.size, self.size), mode="bilinear", align_corners=False).flatten(1).float()
        raw = self.parameters_head(z).permute(0, 2, 3, 1).reshape(b, -1, self.k_max, 11).float()
        allocation_score = self.score(z).flatten(1).float() + self.gamma*p.detach()
        extra = torch.sigmoid((allocation_score[..., None]-self.ordered_thresholds()) / self.cfg["gate_temperature"])
        probability = torch.cat([torch.ones_like(p[..., None]), extra], -1)
        if not self.cfg["iga_enabled"]:
            probability = torch.ones_like(probability)
        hard = probability >= self.cfg["hard_threshold"]
        st = straight_through(hard, probability)
        # A geometric anchor prior is an explicit implementation choice, not specified by the paper.
        bases = []
        with torch.no_grad():
            fractions = (torch.arange(self.k_max, device=z.device).float()+.5)/self.k_max
            for camera in target_cameras:
                origin, direction, _ = camera.rays(self.size, self.size, z.device)
                near, far, valid = ray_box_interval(origin, direction, self.cfg["scene_bound"])
                closest = -(origin*direction).sum(-1)
                near = torch.where(valid, near, closest-.5)
                far = torch.where(valid, far, closest+.5)
                distances = near[:, None] + fractions[None]*(far-near)[:, None]
                center = origin[:, None] + distances[..., None]*direction[:, None]
                bases.append(center.clamp(-self.cfg["scene_bound"], self.cfg["scene_bound"]))
        centers = (torch.stack(bases) + self.cfg["center_offset"]*torch.tanh(raw[..., :3])).clamp(-self.cfg["scene_bound"], self.cfg["scene_bound"])
        quaternion = F.normalize(raw[..., 3:7], dim=-1, eps=1e-8)
        # A zero predicted quaternion has a well-defined identity fallback.
        identity = torch.zeros_like(quaternion)
        identity[..., 0] = 1
        quaternion = torch.where(raw[..., 3:7].norm(dim=-1, keepdim=True) < 1e-8, identity, quaternion)
        scale = self.cfg["scale_min"] + (self.cfg["scale_max"]-self.cfg["scale_min"])*torch.sigmoid(raw[..., 7:10])
        kappa = F.softplus(raw[..., 10])
        gaussians = GaussianSet(centers.flatten(1, 2), scale.flatten(1, 2), quaternion.flatten(1, 2), kappa.flatten(1))
        return Allocation(gaussians, probability, hard, st, p)
