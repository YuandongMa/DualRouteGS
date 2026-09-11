import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from dualroutegs.config import load_config
from dualroutegs.geometry import Camera
from dualroutegs.gaussians import GaussianSet
from dualroutegs.models.psr import PhaseReliability, PhaseAwareStateRouting
from dualroutegs.models.scan import affine_scan, compact_subspace_scan
from dualroutegs.models.iga import InnovationGuidedAllocator
from dualroutegs.rendering.reference import ReferenceRenderer, integrate_chunk
from dualroutegs.diffusion import DiffusionSchedule, reconstruct
from dualroutegs.runtime import build
from dualroutegs.losses import reconstruction_loss
from dualroutegs.data import ProjectionDataset


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class CoreContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.cfg = load_config(ROOT/"configs/smoke.yaml")

    def test_monotonic_phase_reliability(self):
        module = PhaseReliability()
        x = torch.linspace(-20, 20, 51, requires_grad=True)
        chi = module(x)
        self.assertTrue(torch.all(chi[1:] >= chi[:-1]))
        gradient = torch.autograd.grad(chi.sum(), x)[0]
        self.assertTrue(torch.all(gradient >= 0))

    def test_affine_scan_matches_sequential_and_gradients(self):
        a = torch.rand(2, 13, 8, dtype=torch.float64, requires_grad=True)
        b = torch.randn(2, 13, 8, dtype=torch.float64, requires_grad=True)
        history, state = [], torch.zeros(2, 8, dtype=torch.float64)
        for i in range(13):
            state = a[:, i]*state+b[:, i]
            history.append(state)
        serial = torch.stack(history, 1)
        parallel = affine_scan(a, b)
        torch.testing.assert_close(parallel, serial)
        expected = torch.autograd.grad(serial.square().sum(), [a, b], retain_graph=True)
        actual = torch.autograd.grad(parallel.square().sum(), [a, b])
        for x, y in zip(expected, actual):
            torch.testing.assert_close(x, y)

    def test_compact_scan_preserves_inactive_state_and_order(self):
        a, b = torch.rand(2, 17, 12), torch.randn(2, 17, 12)
        ranks = torch.randint(1, 5, (2, 17))
        mask = torch.arange(4)[None, None] < ranks[..., None]
        full = mask.repeat_interleave(3, -1)
        expected = affine_scan(1+full*(a-1), full*b)
        actual = compact_subspace_scan(a, b, mask, 3)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_all_inactive_compact_scan(self):
        result = compact_subspace_scan(torch.ones(1, 7, 8), torch.randn(1, 7, 8), torch.zeros(1, 7, 2, dtype=torch.bool), 4)
        self.assertEqual(float(result.abs().sum()), 0)

    def test_nested_allocation_and_minimum_capacity(self):
        allocator = InnovationGuidedAllocator(self.cfg["model"])
        with torch.no_grad():
            allocator.score.bias.fill_(10)
        output = allocator(torch.randn(2, 32, 4, 4), torch.rand(2, 4, 4), [Camera.circular(), Camera.circular(30)])
        self.assertTrue(torch.all(output.probability[..., 0] == 1))
        self.assertTrue(torch.all(output.probability[..., 1:] <= output.probability[..., :-1]))
        self.assertTrue(torch.all(output.hard[..., 1:] <= output.hard[..., :-1]))
        self.assertTrue(torch.all(output.hard.sum(-1) >= 1))
        self.assertTrue(torch.all(torch.linalg.eigvalsh(output.gaussians.covariance()) > 0))

    def test_gate_gradient_with_zero_forward_attenuation(self):
        allocator = InnovationGuidedAllocator(self.cfg["model"])
        with torch.no_grad():
            allocator.score.weight.zero_()
            allocator.score.bias.fill_(-1)
            allocator.gamma.zero_()
        output = allocator(torch.randn(1, 32, 4, 4), torch.rand(1, 4, 4), [Camera.circular()])
        self.assertFalse(bool(output.hard[..., 1:].any()))
        image = ReferenceRenderer(checkpoint_chunks=False)(output.relaxed(), [Camera.circular()], 8, 8)
        image.sum().backward()
        self.assertGreater(float(allocator.score.bias.grad.abs()), 0)
        self.assertTrue(torch.isfinite(allocator.parameters_head.weight.grad).all())

    def test_reference_ray_matches_analytic_isotropic_gaussian(self):
        sigma, kappa = .2, 1.7
        g = GaussianSet(torch.zeros(1, 1, 3, dtype=torch.float64), torch.full((1, 1, 3), sigma, dtype=torch.float64), torch.tensor([[[1., 0, 0, 0]]], dtype=torch.float64), torch.tensor([[kappa]], dtype=torch.float64))
        value = ReferenceRenderer()(g, [Camera.circular()], 1, 1)[0, 0, 0, 0]
        self.assertAlmostEqual(float(value), kappa*sigma*math.sqrt(2*math.pi), places=9)

    def test_radiative_linearity(self):
        g = GaussianSet(torch.zeros(1, 1, 3), torch.full((1, 1, 3), .2), torch.tensor([[[1., 0, 0, 0]]]), torch.ones(1, 1))
        renderer = ReferenceRenderer(checkpoint_chunks=False)
        a = renderer(g, [Camera.circular()], 9, 9)
        b = renderer(g.weighted(torch.full((1, 1), 3.0)), [Camera.circular()], 9, 9)
        torch.testing.assert_close(b, 3*a)

    def test_integrator_gradcheck(self):
        means = torch.tensor([[.02, .04, .03]], dtype=torch.float64, requires_grad=True)
        precision = torch.eye(3, dtype=torch.float64)[None]*12
        kappa = torch.tensor([.8], dtype=torch.float64, requires_grad=True)
        origin = torch.tensor([[-2., .1, .1]], dtype=torch.float64)
        direction = torch.tensor([[1., 0, 0]], dtype=torch.float64)
        length = torch.tensor([4.], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda m, k: integrate_chunk(m, precision, k, origin, direction, length), (means, kappa)))

    def test_ddim_formula_and_terminal_boundary(self):
        schedule = DiffusionSchedule()
        x0, noise = torch.rand(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
        t, previous = torch.tensor([999, 500]), torch.tensor([700, 200])
        noisy = schedule.q_sample(x0, t, noise)
        actual = schedule.ddim_step(noisy, x0, t, previous)
        expected = schedule.q_sample(x0, previous, noise)
        torch.testing.assert_close(actual, expected)
        terminal = schedule.ddim_step(noisy, x0, t, torch.full_like(t, -1))
        torch.testing.assert_close(terminal, x0)

    def test_empty_query_route_is_identity(self):
        cfg = self.cfg["model"].copy()
        cfg.update(query_threshold=1.0, query_phase_shift=0.0)
        psr = PhaseAwareStateRouting(cfg).eval()
        x = torch.randn(1, 9, cfg["dim"])
        with torch.no_grad():
            result, _, diag = psr(x, torch.tensor([-10.0]))
        torch.testing.assert_close(result, x)
        self.assertEqual(float(diag["query_fraction"][0]), 0)

    def test_full_training_step_has_finite_psr_and_iga_gradients(self):
        model, renderer, schedule = build(self.cfg, "cpu")
        measured, clean = torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16)
        t = torch.tensor([300])
        allocation, _ = model(schedule.q_sample(clean, t, torch.randn_like(clean)), measured, [Camera.circular(90)], [Camera.circular()], schedule.log_snr(t))
        image = renderer(allocation.relaxed(), [Camera.circular(90)], 16, 16)
        novel = renderer(allocation.relaxed(), [Camera.circular(30)], 16, 16)
        loss, _ = reconstruction_loss(image, clean, novel, clean, allocation, self.cfg["loss"])
        loss.backward()
        for name in ["blocks.0.operation.reliability.raw_slope", "blocks.0.operation.A_log", "allocator.score.weight", "allocator.raw_gaps", "allocator.parameters_head.weight"]:
            grad = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(float(grad.abs().sum()), 0, name)

    def test_sampling_is_repeatable_and_resets_memory(self):
        model, renderer, schedule = build(self.cfg, "cpu")
        measured = torch.rand(1, 1, 16, 16)
        a, _, _ = reconstruct(model, renderer, schedule, measured, [Camera.circular()], [Camera.circular(90)], 3, 123)
        reconstruct(model, renderer, schedule, measured, [Camera.circular()], [Camera.circular(90)], 3, 456)
        b, _, _ = reconstruct(model, renderer, schedule, measured, [Camera.circular()], [Camera.circular(90)], 3, 123)
        torch.testing.assert_close(a[0].means, b[0].means, rtol=0, atol=0)
        torch.testing.assert_close(a[0].kappa, b[0].kappa, rtol=0, atol=0)

    def test_patient_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"manifest.json"
            path.write_text(json.dumps({"scenes": [{"id": "a", "group_id": "patient", "split": "train"}, {"id": "b", "group_id": "patient", "split": "test"}]}))
            with self.assertRaisesRegex(ValueError, "leakage"):
                ProjectionDataset(path)

    def test_dense_training_forward_matches_sparse_inference(self):
        psr = PhaseAwareStateRouting(self.cfg["model"])
        x = torch.randn(2, 23, 32)
        log_snr = torch.tensor([-2., 2.])
        with torch.no_grad():
            dense = psr.train()(x, log_snr)[0]
            sparse = psr.eval()(x, log_snr)[0]
        torch.testing.assert_close(dense, sparse, rtol=2e-5, atol=2e-6)

    def test_gradient_checkpointing_path(self):
        self.cfg["model"]["gradient_checkpointing"] = True
        model, _, schedule = build(self.cfg, "cpu")
        measured = torch.rand(1, 1, 16, 16)
        allocation, _ = model(torch.randn_like(measured), measured, [Camera.circular(90)], [Camera.circular()], schedule.log_snr(torch.tensor([700])))
        objective = allocation.gaussians.kappa.mean()+allocation.probability.mean()
        objective.backward()
        self.assertTrue(torch.isfinite(model.target_encoder.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
