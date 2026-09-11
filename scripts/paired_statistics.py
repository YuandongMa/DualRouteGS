"""Group-level paired Wilcoxon tests with Holm correction for multiple metrics."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon


def group_rows(rows, metric):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row[metric])
    return {key: float(np.mean(values)) for key, values in groups.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--metrics", nargs="+", default=["psnr", "ssim"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ours, control = [json.loads(Path(p).read_text())["rows"] for p in [args.ours, args.control]]
    result = []
    for metric in args.metrics:
        a, b = group_rows(ours, metric), group_rows(control, metric)
        if a.keys() != b.keys():
            raise ValueError("Paired testing requires the same independent patient/object set")
        keys = sorted(a)
        difference = np.array([a[k]-b[k] for k in keys])
        p = 1.0 if np.allclose(difference, 0) else float(wilcoxon(difference, alternative="two-sided", zero_method="wilcox").pvalue)
        result.append({"metric": metric, "n": len(keys), "mean_difference": float(difference.mean()), "p_raw": p})
    order = np.argsort([x["p_raw"] for x in result])
    previous = 0.0
    for rank, i in enumerate(order):
        corrected = min(1.0, max(previous, (len(result)-rank)*result[i]["p_raw"]))
        result[i]["p_holm"] = corrected
        previous = corrected
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
