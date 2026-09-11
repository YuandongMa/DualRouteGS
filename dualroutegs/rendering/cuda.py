"""Narrow adapter to the bundled radiative CUDA backend.

The backend is an affine ray-space splatter, not the exact finite-ray oracle.
Original backend licensing and headers are retained under third_party/.
"""
import torch


def packed_covariance(g):
    c = g.covariance()
    return torch.stack([c[..., 0, 0], c[..., 0, 1], c[..., 0, 2], c[..., 1, 1], c[..., 1, 2], c[..., 2, 2]], -1)


class CudaRenderer:
    def __init__(self):
        try:
            import xray_gaussian_rasterization_voxelization as extension
        except ImportError as exc:
            raise RuntimeError("Build the bundled backend: python scripts/install_backend.py") from exc
        if not getattr(extension, "DUALROUTEGS_GATE_SAFE", False):
            raise RuntimeError("The installed backend lacks the zero-density gate-gradient patch. Reinstall third_party/ray_gaussian.")
        self.extension = extension

    def __call__(self, gaussians, cameras, height, width):
        if not gaussians.means.is_cuda:
            raise ValueError("CUDA renderer requires CUDA tensors")
        if len(cameras) != gaussians.means.shape[0]:
            raise ValueError("One camera per batch item required")
        with torch.autocast("cuda", enabled=False):
            g = gaussians.to(dtype=torch.float32)
            covariance = packed_covariance(g)
            output = []
            for bi, camera in enumerate(cameras):
                if camera.beam != "cone" or max(abs(x) for x in camera.offset) > 1e-8:
                    raise ValueError("The CUDA adapter currently supports centered cone-beam cameras. Use reference for parallel/offset geometry.")
                c2w = torch.tensor(camera.c2w, dtype=torch.float32, device=g.means.device)
                view = torch.linalg.inv(c2w).T.contiguous()
                tan_x = camera.detector_size[1]/(2*camera.dsd)
                tan_y = camera.detector_size[0]/(2*camera.dsd)
                near, far = .01, 100.0
                projection = torch.zeros(4, 4, device=g.means.device)
                projection[0, 0], projection[1, 1] = 1/tan_x, 1/tan_y
                projection[2, 2] = far/(far-near)
                projection[2, 3] = -far*near/(far-near)
                projection[3, 2] = 1
                settings = self.extension.GaussianRasterizationSettings(
                    image_height=height, image_width=width, tanfovx=tan_x, tanfovy=tan_y,
                    scale_modifier=1.0, viewmatrix=view, projmatrix=(view@projection.T).contiguous(),
                    campos=c2w[:3, 3].contiguous(), prefiltered=False, mode=1, debug=False)
                rasterizer = self.extension.GaussianRasterizer(raster_settings=settings)
                means = g.means[bi].contiguous()
                image, _ = rasterizer(means3D=means, means2D=torch.zeros_like(means),
                                      opacities=g.kappa[bi, :, None].contiguous(), cov3D_precomp=covariance[bi].contiguous())
                output.append(image)
            return torch.stack(output)

    def voxelize(self, gaussians, resolution=256, bound=1.0):
        if not gaussians.means.is_cuda:
            raise ValueError("CUDA voxelization requires CUDA tensors")
        g = gaussians.to(dtype=torch.float32)
        covariance = packed_covariance(g)
        settings = self.extension.GaussianVoxelizationSettings(
            scale_modifier=1.0, nVoxel_x=resolution, nVoxel_y=resolution, nVoxel_z=resolution,
            sVoxel_x=2*bound, sVoxel_y=2*bound, sVoxel_z=2*bound,
            center_x=0.0, center_y=0.0, center_z=0.0, prefiltered=False, debug=False)
        voxelizer = self.extension.GaussianVoxelizer(voxel_settings=settings)
        result = []
        for bi in range(g.means.shape[0]):
            volume, _ = voxelizer(means3D=g.means[bi].contiguous(), opacities=g.kappa[bi, :, None].contiguous(), cov3D_precomp=covariance[bi].contiguous())
            result.append(volume.squeeze(0) if volume.ndim == 4 else volume)
        return torch.stack(result)
