from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import yaml


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"Cyclic config inheritance: {path}")
    seen.add(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = data.pop("_base_", None)
    return merge(load_config(path.parent / base, seen), data) if base else data


def validate_config(cfg):
    m = cfg["model"]
    if m["dim"] % 4 or m["dim"] % m["heads"]:
        raise ValueError("dim must be divisible by 4 and heads")
    if m["state_dim"] % m["subspaces"]:
        raise ValueError("state_dim must be divisible by subspaces")
    if not 1 <= m["r_min"] <= m["subspaces"]:
        raise ValueError("Require 1 <= r_min <= subspaces")
    if cfg["data"]["image_size"] % m["patch_size"]:
        raise ValueError("image_size must be divisible by patch_size")
    if m["k_max"] < 1 or m["gate_temperature"] <= 0:
        raise ValueError("Invalid allocation bank or temperature")
    if not 0 < m["hard_threshold"] < 1:
        raise ValueError("hard_threshold must be strictly between 0 and 1")
    if m["interaction"] not in {"psr", "static_state", "attention"}:
        raise ValueError("Unknown interaction")
    if cfg["renderer"]["backend"] not in {"reference", "cuda"}:
        raise ValueError("Choose reference or cuda renderer explicitly")
    if cfg["data"]["projection_scale"] <= 0:
        raise ValueError("projection_scale must be positive")
    if cfg["train"]["accumulation"] < 1 or cfg["train"]["steps"] < 1:
        raise ValueError("Invalid training steps or accumulation")
    return cfg
