"""Create tiny analytic Gaussian phantoms for pipeline validation, not benchmarks."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from dualroutegs.geometry import Camera
from dualroutegs.gaussians import GaussianSet
from dualroutegs.rendering.reference import ReferenceRenderer, voxelize_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/demo")
    parser.add_argument("--size", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    renderer = ReferenceRenderer(checkpoint_chunks=False)
    cameras = [Camera.circular(angle, detector_size=(4.8, 4.8)) for angle in [0, 30, 60, 90, 120, 150]]
    canonical = Camera.circular(90)
    records = []
    for index, split in enumerate(["train", "val", "test"]):
        center = torch.tensor([[[-.32+.03*index, .1, .15], [.3, -.2, -.2], [0., .35, -.3]]])
        scales = torch.tensor([[[.18, .13, .25], [.12, .22, .13], [.07, .08, .2]]])
        rotations = torch.tensor([[[1., 0, 0, 0]]]).expand(1, 3, 4).clone()
        phantom = GaussianSet(center, scales, rotations, torch.tensor([[.8, 1.2, .6]]))
        with torch.no_grad():
            images = torch.cat([renderer(phantom, [c], args.size, args.size) for c in cameras])[:, 0].numpy()
            volume = voxelize_reference(phantom, 16)[0].numpy()
        name = f"phantom_{split}"
        np.savez_compressed(output/f"{name}.npz", projections=images)
        np.save(output/f"{name}_volume.npy", volume)
        geometry = {"views": [c.to_dict() for c in cameras], "canonical": canonical.to_dict()}
        (output/f"{name}_geometry.json").write_text(json.dumps(geometry, indent=2))
        records.append({"id": name, "group_id": name, "domain": "synthetic_phantom", "split": split, "projections": f"{name}.npz", "geometry": f"{name}_geometry.json", "volume": f"{name}_volume.npy", "input_index": 0, "target_indices": [1, 2, 3, 4, 5]})
        if split == "test":
            np.save(output/"example_input.npy", images[0])
            (output/"example_camera.json").write_text(json.dumps({"condition": cameras[0].to_dict(), "canonical": canonical.to_dict()}, indent=2))
    (output/"manifest.json").write_text(json.dumps({"format_version": 1, "description": "Synthetic validation phantoms only", "scenes": records}, indent=2))
    print(output/"manifest.json")


if __name__ == "__main__":
    main()
