from pathlib import Path
import random
import numpy as np
import torch
from .models import DualRouteGS
from .diffusion import DiffusionSchedule
from .rendering import make_renderer
from .config import validate_config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build(cfg, device):
    validate_config(cfg)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use the smoke/reference config on CPU")
    model = DualRouteGS(cfg["model"]).to(device)
    return model, make_renderer(cfg["renderer"]), DiffusionSchedule(**cfg["diffusion"])


def save_checkpoint(path, model, optimizer, step, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": cfg,
               "torch_rng": torch.get_rng_state(), "python_rng": random.getstate()}
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    # Tensor/state-only loading avoids arbitrary pickle execution.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return checkpoint
