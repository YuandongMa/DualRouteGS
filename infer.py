import argparse
import json
from pathlib import Path
import numpy as np
import torch
from dualroutegs.data import load_projection
from dualroutegs.geometry import Camera
from dualroutegs.runtime import load_checkpoint, build
from dualroutegs.diffusion import reconstruct
from dualroutegs.rendering.reference import voxelize_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Single scalar log-attenuation .npy")
    parser.add_argument("--camera", required=True, help="JSON with condition and canonical calibrated cameras")
    parser.add_argument("--output", default="outputs/reconstruction")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", choices=["reference", "cuda"])
    parser.add_argument("--volume-resolution", type=int, default=0)
    args = parser.parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    cfg = ckpt["config"]
    if args.backend:
        cfg["renderer"]["backend"] = args.backend
    torch.set_num_threads(cfg["train"].get("cpu_threads", 4))
    model, renderer, schedule = build(cfg, args.device)
    model.load_state_dict(ckpt["model"], strict=True)
    cameras = json.loads(Path(args.camera).read_text(encoding="utf-8"))
    condition = Camera(**cameras["condition"])
    canonical = Camera(**cameras["canonical"])
    image = load_projection(args.input, cfg["data"]["image_size"])[None].to(args.device) / cfg["data"]["projection_scale"]
    with torch.inference_mode():
        active, canvas, trace = reconstruct(model, renderer, schedule, image, [condition], [canonical], args.steps or cfg["inference"]["steps"], args.seed)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output/"gaussians.npz", **active[0].numpy_dict())
        np.save(output/"canonical_projection.npy", canvas[0, 0].cpu().numpy()*cfg["data"]["projection_scale"])
        measured_render = renderer(active[0], [condition], *image.shape[-2:])
        np.save(output/"condition_projection.npy", measured_render[0, 0].cpu().numpy()*cfg["data"]["projection_scale"])
        if args.volume_resolution:
            if hasattr(renderer, "voxelize"):
                volume = renderer.voxelize(active[0], args.volume_resolution, cfg["model"]["scene_bound"])
            else:
                volume = voxelize_reference(active[0], args.volume_resolution, cfg["model"]["scene_bound"])
            np.save(output/"volume.npy", volume[0].cpu().numpy())
    metadata = {"coordinate_system": "global scanner; voxel array axes xyz", "scene_bound": cfg["model"]["scene_bound"], "projection_scale": cfg["data"]["projection_scale"], "checkpoint_step": ckpt["step"], "seed": args.seed, "backend": cfg["renderer"]["backend"], "trace": trace}
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(active[0].kappa[0])} Gaussians to {output}")


if __name__ == "__main__":
    main()
