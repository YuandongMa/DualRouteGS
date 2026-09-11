"""Image-space forward diffusion and deterministic DDIM from rendered x0."""
import math
import torch


class DiffusionSchedule:
    def __init__(self, timesteps=1000, schedule="cosine"):
        self.timesteps = timesteps
        if timesteps < 2:
            raise ValueError("At least two training timesteps required")
        if schedule == "cosine":
            s = torch.linspace(0, 1, timesteps+1, dtype=torch.float64)
            cumulative = torch.cos((s+.008)/1.008*math.pi/2).square()
            beta = (1-cumulative[1:]/cumulative[:-1]).clamp(.00001, .999)
        elif schedule == "linear":
            beta = torch.linspace(.0001, .02, timesteps, dtype=torch.float64)
        else:
            raise ValueError("Unknown diffusion schedule")
        self.alpha_bar = (1-beta).cumprod(0).float()

    def alpha(self, t):
        return self.alpha_bar.to(t.device)[t]

    def log_snr(self, t):
        alpha = self.alpha(t).clamp(1e-8, 1-1e-7)
        return torch.log(alpha)-torch.log1p(-alpha)

    def q_sample(self, clean, t, noise):
        alpha = self.alpha(t).reshape(-1, 1, 1, 1)
        return alpha.sqrt()*clean + (1-alpha).sqrt()*noise

    def ddim_step(self, noisy, x0, t, previous):
        alpha = self.alpha(t).reshape(-1, 1, 1, 1)
        safe_previous = previous.clamp_min(0)
        next_alpha = torch.where(previous < 0, torch.ones_like(previous, dtype=torch.float32), self.alpha(safe_previous)).reshape(-1, 1, 1, 1)
        epsilon = (noisy-alpha.sqrt()*x0) / (1-alpha).sqrt().clamp_min(1e-8)
        return next_alpha.sqrt()*x0 + (1-next_alpha).sqrt()*epsilon

    def inference_timesteps(self, steps):
        if not 2 <= steps <= self.timesteps:
            raise ValueError("DDIM steps must be between 2 and training timesteps")
        return torch.linspace(self.timesteps-1, 0, steps).round().long().tolist()


@torch.no_grad()
def reconstruct(model, renderer, schedule, measured, condition_cameras, canonical_cameras, steps=20, seed=0):
    """The only measured input is `measured`; no held-out pixels enter sampling."""
    previous_mode = model.training
    model.eval()
    try:
        generator = torch.Generator(device=measured.device).manual_seed(seed)
        canvas = torch.randn(measured.shape, generator=generator, device=measured.device)
        timeline = schedule.inference_timesteps(steps)
        trace = []
        for index, value in enumerate(timeline):
            t = torch.full((measured.shape[0],), value, dtype=torch.long, device=measured.device)
            allocation, diagnostic = model(canvas, measured, canonical_cameras, condition_cameras, schedule.log_snr(t))
            active = allocation.active()
            x0 = torch.cat([renderer(g, [camera], *measured.shape[-2:]) for g, camera in zip(active, canonical_cameras)], 0)
            next_t = timeline[index+1] if index+1 < len(timeline) else -1
            canvas = schedule.ddim_step(canvas, x0, t, torch.full_like(t, next_t))
            trace.append({"t": value, "active_gaussians": [int(g.means.shape[1]) for g in active], "query_fraction": float(torch.stack([d["query_fraction"].mean() for d in diagnostic]).mean()), "active_subspaces": float(torch.stack([d["active_subspaces"].float().mean() for d in diagnostic]).mean())})
        return active, canvas, trace
    finally:
        model.train(previous_mode)
