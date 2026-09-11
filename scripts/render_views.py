"""Render any supplied calibrated views from exported attenuation Gaussians."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from dualroutegs.gaussians import GaussianSet
from dualroutegs.geometry import Camera
from dualroutegs.rendering import make_renderer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussians", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--backend", choices=["reference", "cuda"], default="cuda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--projection-scale", type=float, default=1.0)
    args = parser.parse_args()
    with np.load(args.gaussians, allow_pickle=False) as data:
        gaussians = GaussianSet(**{key: torch.from_numpy(data[key]).to(args.device) for key in ["means", "scales", "rotations", "kappa"]})
    cameras = [Camera(**camera) for camera in json.loads(Path(args.geometry).read_text())["views"]]
    renderer = make_renderer({"backend": args.backend})
    with torch.inference_mode():
        images = [renderer(gaussians, [camera], args.size, args.size)[0, 0].cpu().numpy()*args.projection_scale for camera in cameras]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, projections=np.stack(images))
