"""Calibrated scanner geometry. Camera axes: right, down, forward (+z)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch


@dataclass
class Camera:
    c2w: list
    detector_size: list  # [height, width], in normalized world units
    dsd: float
    beam: str = "cone"
    offset: tuple = (0.0, 0.0)  # [vertical, horizontal] in detector units

    def __post_init__(self):
        mat = np.asarray(self.c2w, dtype=float)
        if mat.shape != (4, 4) or not np.isfinite(mat).all():
            raise ValueError("c2w must be a finite 4x4 matrix")
        rot = mat[:3, :3]
        if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rot), 1, atol=1e-4):
            raise ValueError("c2w rotation must be orthonormal and right handed")
        if not np.allclose(mat[3], [0, 0, 0, 1]):
            raise ValueError("Invalid homogeneous transform")
        if self.beam not in {"cone", "parallel"}:
            raise ValueError("beam must be cone or parallel")
        if self.dsd <= 0 or len(self.detector_size) != 2 or min(self.detector_size) <= 0:
            raise ValueError("Positive detector dimensions and source-detector distance required")
        if len(self.offset) != 2 or not np.isfinite([self.dsd, *self.detector_size, *self.offset]).all():
            raise ValueError("Camera dimensions and offsets must be finite")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def circular(cls, angle_deg=0.0, dso=3.0, dsd=6.0, detector_size=(4.8, 4.8), beam="cone"):
        angle = math.radians(angle_deg)
        # Matches the circular CT camera convention of the supplied scanner code.
        right = [-math.sin(angle), math.cos(angle), 0.0]
        down = [0.0, 0.0, -1.0]
        forward = [-math.cos(angle), -math.sin(angle), 0.0]
        mat = np.eye(4)
        mat[:3, :3] = np.array([right, down, forward]).T
        mat[:3, 3] = [dso * math.cos(angle), dso * math.sin(angle), 0]
        return cls(mat.tolist(), list(detector_size), dsd, beam)

    def rays(self, height, width, device="cpu", dtype=torch.float32):
        mat = torch.as_tensor(self.c2w, dtype=dtype, device=device)
        v = ((torch.arange(height, device=device, dtype=dtype) + .5) / height - .5) * self.detector_size[0] + self.offset[0]
        u = ((torch.arange(width, device=device, dtype=dtype) + .5) / width - .5) * self.detector_size[1] + self.offset[1]
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        lateral = uu[..., None] * mat[:3, 0] + vv[..., None] * mat[:3, 1]
        if self.beam == "cone":
            origins = mat[:3, 3].expand(height, width, 3)
            vectors = lateral + self.dsd * mat[:3, 2]
            lengths = vectors.norm(dim=-1)
            directions = vectors / lengths[..., None]
        else:
            origins = mat[:3, 3] + lateral
            directions = mat[:3, 2].expand(height, width, 3)
            lengths = torch.full((height, width), self.dsd, device=device, dtype=dtype)
        return origins.reshape(-1, 3), directions.reshape(-1, 3), lengths.reshape(-1)


def ray_features(cameras, height, width, device, dtype):
    result = []
    for camera in cameras:
        origin, direction, _ = camera.rays(height, width, device, dtype)
        moment = torch.linalg.cross(origin, direction, dim=-1)
        result.append(torch.cat([moment, direction], -1).T.reshape(6, height, width))
    return torch.stack(result)


def ray_box_interval(origin, direction, bound=1.0):
    parallel = direction.abs() < 1e-8
    safe = torch.where(parallel, torch.ones_like(direction), direction)
    a, b = (-bound - origin) / safe, (bound - origin) / safe
    lo, hi = torch.minimum(a, b), torch.maximum(a, b)
    outside = parallel & (origin.abs() > bound)
    lo = torch.where(parallel, torch.full_like(lo, -torch.inf), lo)
    hi = torch.where(parallel, torch.full_like(hi, torch.inf), hi)
    near = lo.amax(-1).clamp_min(0)
    far = hi.amin(-1)
    valid = (far > near) & ~outside.any(-1)
    return near, far, valid
