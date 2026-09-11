"""Required on the user's CUDA machine before a full-scale experiment."""
import unittest
import torch
from dualroutegs.geometry import Camera
from dualroutegs.gaussians import GaussianSet
from dualroutegs.rendering.reference import ReferenceRenderer, voxelize_reference


@unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware unavailable")
class CudaContracts(unittest.TestCase):
    def setUp(self):
        from dualroutegs.rendering.cuda import CudaRenderer
        self.renderer = CudaRenderer()

    def test_zero_density_gate_gradient(self):
        density = torch.zeros(1, 1, device="cuda", requires_grad=True)
        g = GaussianSet(torch.zeros(1, 1, 3, device="cuda"), torch.full((1, 1, 3), .15, device="cuda"), torch.tensor([[[1., 0, 0, 0]]], device="cuda"), density)
        image = self.renderer(g, [Camera.circular()], 32, 32)
        self.assertEqual(float(image.abs().sum()), 0)
        image.sum().backward()
        self.assertTrue(torch.isfinite(density.grad).all())
        self.assertGreater(float(density.grad[0, 0]), 0)

    def test_small_central_gaussian_reference_agreement(self):
        g = GaussianSet(torch.zeros(1, 1, 3, device="cuda"), torch.full((1, 1, 3), .08, device="cuda"), torch.tensor([[[1., 0, 0, 0]]], device="cuda"), torch.ones(1, 1, device="cuda"))
        with torch.no_grad():
            expected = ReferenceRenderer(checkpoint_chunks=False)(g, [Camera.circular()], 33, 33)
            actual = self.renderer(g, [Camera.circular()], 33, 33)
        # Affine projected splatting has a documented approximation, especially off-axis.
        self.assertLess(float((actual-expected).abs().sum()/expected.abs().sum()), .10)

    def test_voxelization_xyz_and_support(self):
        g = GaussianSet(torch.tensor([[[.3, -.2, .1]]], device="cuda"), torch.tensor([[[.18, .12, .1]]], device="cuda"), torch.tensor([[[1., 0, 0, 0]]], device="cuda"), torch.ones(1, 1, device="cuda"))
        with torch.no_grad():
            expected = voxelize_reference(g, 32)
            actual = self.renderer.voxelize(g, 32)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)
