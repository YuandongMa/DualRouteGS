"""Fixed-canvas single-view reconstruction, then held-out view/volume metrics."""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from dualroutegs.runtime import load_checkpoint, build
from dualroutegs.data import ProjectionDataset
from dualroutegs.diffusion import reconstruct
from dualroutegs.metrics import projection_metrics, volume_metrics, bootstrap_mean
from dualroutegs.rendering.reference import voxelize_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="outputs/evaluation")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--volume", action="store_true")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--volume-data-range", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    cfg = ckpt["config"]
    torch.set_num_threads(cfg["train"].get("cpu_threads", 4))
    model, renderer, schedule = build(cfg, args.device)
    model.load_state_dict(ckpt["model"], strict=True)
    dataset = ProjectionDataset(args.manifest or cfg["data"]["manifest"], args.split, cfg["data"]["image_size"], cfg["data"]["projection_scale"])
    lpips_net = None
    if args.lpips:
        import lpips
        lpips_net = lpips.LPIPS(net="alex").to(args.device).eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(len(dataset)):
        record, images, cameras, observed, targets, canonical = dataset.read_scene(index)
        condition = images[observed:observed+1].to(args.device)
        # Warm up this shape before measuring synchronized sampling latency.
        with torch.inference_mode():
            if index == 0:
                reconstruct(model, renderer, schedule, condition, [cameras[observed]], [canonical], args.steps or cfg["inference"]["steps"], args.seed)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            active, _, trace = reconstruct(model, renderer, schedule, condition, [cameras[observed]], [canonical], args.steps or cfg["inference"]["steps"], args.seed+index)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            peak_memory = torch.cuda.max_memory_allocated()/2**30 if args.device.startswith("cuda") else None
            view_rows = []
            for target in targets:
                # Target pixels are used only below for metrics, after the Gaussian field is fixed.
                pred = renderer(active[0], [cameras[target]], *condition.shape[-2:])
                truth = images[target:target+1].to(args.device)
                metrics = projection_metrics(pred, truth, cfg["loss"]["data_range"])
                if lpips_net:
                    if pred.min() < 0 or pred.max() > 1 or truth.min() < 0 or truth.max() > 1:
                        raise ValueError("LPIPS requires a declared [0,1] projection normalization; set projection_scale using train data")
                    metrics["lpips"] = float(lpips_net((pred*2-1).repeat(1, 3, 1, 1), (truth*2-1).repeat(1, 3, 1, 1)).mean())
                view_rows.append(metrics)
            row = {"id": record["id"], "group_id": record["group_id"], "domain": record.get("domain", "unspecified"), "time_seconds": elapsed, "peak_gpu_gib": peak_memory, "active_gaussians": active[0].means.shape[1], "views": len(targets)}
            row.update({k: float(np.mean([v[k] for v in view_rows])) for k in view_rows[0]})
            if args.volume:
                if "volume" not in record:
                    raise ValueError(f"No ground-truth volume for {record['id']}")
                truth = np.load(dataset.root/record["volume"], allow_pickle=False).astype(np.float32)/cfg["data"]["projection_scale"]
                if len(set(truth.shape)) != 1:
                    raise ValueError("Volume must be resampled to the common cubic grid")
                volume = renderer.voxelize(active[0], truth.shape[0], cfg["model"]["scene_bound"]) if hasattr(renderer, "voxelize") else voxelize_reference(active[0], truth.shape[0], cfg["model"]["scene_bound"])
                row.update(volume_metrics(volume[0].cpu().numpy(), truth, args.volume_data_range))
            rows.append(row)
            print(json.dumps(row), flush=True)
    # Average views within scenes, then scenes within a patient/object, then groups.
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    metrics = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k != "views"]
    summary = {key: bootstrap_mean([np.mean([r[key] for r in records]) for records in grouped.values()], args.seed) for key in metrics}
    payload = {"rows": rows, "summary": summary, "timing_scope": "DDIM including Gaussian decoding, hard routing and canonical rendering; excludes data loading and held-out metric rendering", "backend": cfg["renderer"]["backend"], "parameters": sum(p.numel() for p in model.parameters()), "seed": args.seed}
    (output/"metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Metrics: {output/'metrics.json'}")


if __name__ == "__main__":
    main()
