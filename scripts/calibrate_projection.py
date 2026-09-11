"""Flat/dark correction and log transmission; scalar arrays only."""
import argparse
from pathlib import Path
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--flat", required=True)
    parser.add_argument("--dark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()
    if not 0 < args.epsilon < 1:
        raise ValueError("epsilon must be strictly between 0 and 1")
    raw, flat, dark = [np.load(p, allow_pickle=False).astype(np.float64) for p in [args.raw, args.flat, args.dark]]
    if any(x.ndim != 2 for x in [raw, flat, dark]) or raw.shape != flat.shape or raw.shape != dark.shape:
        raise ValueError("Matching scalar 2D raw/flat/dark arrays required")
    if not all(np.isfinite(x).all() for x in [raw, flat, dark]) or (flat-dark <= 0).any():
        raise ValueError("Invalid flat/dark calibration")
    ratio = (raw-dark)/(flat-dark)
    corrected = -np.log(np.clip(ratio, args.epsilon, 1.0))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, corrected.astype(np.float32))
    print(f"Saved log attenuation; clipped pixels: {np.mean((ratio <= args.epsilon) | (ratio > 1)):.6%}")
