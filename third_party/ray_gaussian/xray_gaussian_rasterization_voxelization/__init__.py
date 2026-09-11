from .rasterization import GaussianRasterizationSettings, GaussianRasterizer
from .voxelization import GaussianVoxelizationSettings, GaussianVoxelizer

# DualRouteGS adapter requires the bundled zero-density-gradient kernel patch.
DUALROUTEGS_GATE_SAFE = True
