"""Evaluate provided masks; never infer material/nodule identity from attenuation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from dualroutegs.region_metrics import dice, normalized_surface_dice, cldice, masked_reconstruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--pred-mask")
    parser.add_argument("--true-mask")
    parser.add_argument("--spacing", nargs=3, type=float, default=[1., 1., 1.])
    parser.add_argument("--tolerance", type=float, default=1.)
    parser.add_argument("--data-range", type=float, default=1.)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pred, truth, roi = [np.load(p, allow_pickle=False) for p in [args.prediction, args.truth, args.roi]]
    result = masked_reconstruction(pred, truth, roi, args.data_range)
    if bool(args.pred_mask) != bool(args.true_mask):
        raise ValueError("Both independently defined binary structure masks are required")
    if args.pred_mask:
        pm, tm = [np.load(p, allow_pickle=False).astype(bool) for p in [args.pred_mask, args.true_mask]]
        result.update(dice=dice(pm, tm), nsd_boundary_voxels=normalized_surface_dice(pm, tm, args.spacing, args.tolerance), cldice=cldice(pm, tm))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
