from __future__ import annotations
from dataclasses import dataclass, fields
import torch
import torch.nn.functional as F


def rotation_matrix(quaternion):
    q = F.normalize(quaternion, dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
    ], -1).reshape(*q.shape[:-1], 3, 3)


@dataclass
class GaussianSet:
    means: torch.Tensor       # [B,N,3]
    scales: torch.Tensor      # [B,N,3], positive standard deviations
    rotations: torch.Tensor   # [B,N,4], wxyz unit quaternion
    kappa: torch.Tensor       # [B,N], physical attenuation, no existence encoded

    def covariance(self):
        rotation = rotation_matrix(self.rotations)
        return (rotation * self.scales.square().unsqueeze(-2)) @ rotation.transpose(-1, -2)

    def precision(self):
        rotation = rotation_matrix(self.rotations)
        return (rotation * self.scales.reciprocal().square().unsqueeze(-2)) @ rotation.transpose(-1, -2)

    def to(self, *args, **kwargs):
        return GaussianSet(**{f.name: getattr(self, f.name).to(*args, **kwargs) for f in fields(self)})

    def sample(self, index):
        return GaussianSet(**{f.name: getattr(self, f.name)[index:index+1] for f in fields(self)})

    def select(self, index, mask):
        return GaussianSet(**{f.name: getattr(self, f.name)[index, mask][None] for f in fields(self)})

    def weighted(self, existence):
        return GaussianSet(self.means, self.scales, self.rotations, self.kappa * existence)

    def numpy_dict(self):
        return {f.name: getattr(self, f.name).detach().cpu().numpy() for f in fields(self)}
