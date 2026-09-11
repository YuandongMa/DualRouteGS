"""Train from calibrated multi-view supervision with one measured condition."""
import argparse
import json
import math
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dualroutegs.config import load_config
from dualroutegs.runtime import build, seed_all, save_checkpoint, load_checkpoint
from dualroutegs.data import ProjectionDataset, collate_samples
from dualroutegs.losses import reconstruction_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/development.yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--output", default="runs/dualroutegs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    resumed = load_checkpoint(args.resume) if args.resume else None
    cfg = resumed["config"] if resumed else load_config(args.config)
    if args.manifest:
        cfg["data"]["manifest"] = args.manifest
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    seed_all(cfg["seed"])
    torch.set_num_threads(cfg["train"].get("cpu_threads", 4))
    model, renderer, schedule = build(cfg, args.device)
    data = cfg["data"]
    dataset = ProjectionDataset(data["manifest"], "train", data["image_size"], data["projection_scale"])
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["workers"], collate_fn=collate_samples)
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    first = 0
    if resumed:
        model.load_state_dict(resumed["model"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        first = resumed["step"]
        torch.set_rng_state(resumed["torch_rng"].cpu())
        random.setstate(resumed["python_rng"])
        if "cuda_rng" in resumed and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resumed["cuda_rng"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    accumulation = cfg["train"]["accumulation"]
    model.train()
    amp = cfg["train"]["amp"] and args.device.startswith("cuda")
    if amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This AMP path uses bfloat16; set train.amp=false for this GPU")
    for step in range(first, cfg["train"]["steps"]):
        warmup = cfg["train"]["warmup"]
        if step < warmup:
            ratio = (step+1)/max(warmup, 1)
        else:
            progress = (step-warmup)/max(cfg["train"]["steps"]-warmup, 1)
            ratio = .5*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups:
            group["lr"] = cfg["train"]["lr"]*ratio
        optimizer.zero_grad(set_to_none=True)
        logs = {}
        for _ in range(accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            measured, clean, novel_gt = [batch[key].to(args.device) for key in ["measured", "clean", "novel"]]
            t = torch.randint(schedule.timesteps, (len(clean),), device=args.device)
            noisy = schedule.q_sample(clean, t, torch.randn_like(clean))
            with torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16, enabled=amp):
                allocation, diagnostic = model(noisy, measured, batch["target_camera"], batch["condition_camera"], schedule.log_snr(t))
            gaussians = allocation.relaxed()
            denoised = renderer(gaussians, batch["target_camera"], *clean.shape[-2:])
            novel = renderer(gaussians, batch["novel_camera"], *clean.shape[-2:])
            loss, parts = reconstruction_loss(denoised, clean, novel, novel_gt, allocation, cfg["loss"], cfg["model"]["iga_enabled"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step+1}")
            (loss/accumulation).backward()
            for key, value in parts.items():
                logs[key] = logs.get(key, 0.0) + float(value.detach())/accumulation
            logs["active_gaussians"] = float(allocation.hard.flatten(1).sum(1).float().mean())
            logs["query_fraction"] = float(diagnostic[-1]["query_fraction"].mean())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        logs.update(step=step+1, lr=optimizer.param_groups[0]["lr"], grad_norm=float(grad_norm))
        with (output/"train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(logs)+"\n")
        if (step+1) % cfg["train"]["log_every"] == 0 or step == first:
            print(json.dumps(logs), flush=True)
        if (step+1) % cfg["train"]["save_every"] == 0 or step+1 == cfg["train"]["steps"]:
            save_checkpoint(output/"last.pt", model, optimizer, step+1, cfg)
    print(f"Checkpoint: {output/'last.pt'}")


if __name__ == "__main__":
    main()
