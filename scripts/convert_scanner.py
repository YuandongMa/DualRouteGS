"""Convert calibrated meta_data.json scene folders to DualRouteGS manifests.

Requires an explicit split/group catalogue; never partitions individual views.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from dualroutegs.geometry import Camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, help="JSON list: id, path, group_id, split, domain, input_index")
    parser.add_argument("--output", required=True)
    parser.add_argument("--canonical-angle", type=float, default=90.0)
    args = parser.parse_args()
    source_file = Path(args.catalog).resolve()
    catalogue = json.loads(source_file.read_text())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for item in catalogue:
        path = Path(item["path"])
        path = path if path.is_absolute() else source_file.parent/path
        meta = json.loads((path/"meta_data.json").read_text())
        scanner = meta["scanner"]
        if max(abs(x) for x in scanner.get("offOrigin", [0, 0, 0])) > 1e-8:
            raise ValueError("Nonzero offOrigin needs explicit calibrated c2w conversion; do not silently ignore it")
        size_voxel = np.asarray(scanner["sVoxel"], dtype=float)
        if not np.allclose(size_voxel, size_voxel[0]):
            raise ValueError("Resample/pad volumes to a cubic field of view before conversion")
        scene_scale = 2.0/float(size_voxel.max())
        detector = (np.asarray(scanner["sDetector"])*scene_scale).tolist()
        dso, dsd = float(scanner["DSO"])*scene_scale, float(scanner["DSD"])*scene_scale
        offset = (np.asarray(scanner.get("offDetector", [0, 0]))*scene_scale).tolist()
        beam = scanner["mode"]
        # Existing train/test entries are acquisition views of ONE scene, not subject splits.
        entries = meta["proj_train"]+meta.get("proj_test", [])
        images, cameras, seen = [], [], set()
        for frame in entries:
            key = (frame["file_path"], float(frame["angle"]))
            if key in seen:
                continue
            seen.add(key)
            image = np.load(path/frame["file_path"], allow_pickle=False).astype(np.float32)
            if image.ndim != 2 or not np.isfinite(image).all() or image.min() < -1e-6:
                raise ValueError("Expected nonnegative scalar line-integral projections")
            images.append(image)
            camera = Camera.circular(np.degrees(frame["angle"]), dso, dsd, detector, beam)
            camera.offset = tuple(offset)
            cameras.append(camera)
        canonical = Camera.circular(args.canonical_angle, dso, dsd, detector, beam)
        canonical.offset = tuple(offset)
        name = item["id"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Scene id must be a plain filename component")
        np.savez_compressed(output/f"{name}.npz", projections=np.stack(images))
        (output/f"{name}_geometry.json").write_text(json.dumps({"views": [c.to_dict() for c in cameras], "canonical": canonical.to_dict()}, indent=2))
        record = {k: item[k] for k in ["id", "group_id", "split", "domain", "input_index"]}
        record.update(projections=f"{name}.npz", geometry=f"{name}_geometry.json", target_indices=[i for i in range(len(images)) if i != item["input_index"]], world_scale=scene_scale)
        if meta.get("vol"):
            volume = np.load(path/meta["vol"], allow_pickle=False).astype(np.float32)
            # mu_normalized * dx_normalized = mu_physical * dx_physical.
            np.save(output/f"{name}_volume.npy", volume/scene_scale)
            record["volume"] = f"{name}_volume.npy"
        records.append(record)
    groups = {}
    for record in records:
        previous = groups.setdefault(record["group_id"], record["split"])
        if previous != record["split"]:
            raise ValueError("Patient/object crosses data partitions")
    (output/"manifest.json").write_text(json.dumps({"format_version": 1, "scenes": records}, indent=2))
    print(output/"manifest.json")


if __name__ == "__main__":
    main()
