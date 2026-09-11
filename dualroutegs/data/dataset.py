"""Scene/patient-split calibrated line-integral datasets; no RGB pseudocolor input."""
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from ..geometry import Camera


def load_projection(path, expected_size=None):
    array = np.load(path, allow_pickle=False)
    if isinstance(array, np.lib.npyio.NpzFile):
        with array as archive:
            array = archive["projection"].copy()
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError("Expected a scalar [H,W] log-attenuation projection; RGB pseudocolor is not physical attenuation")
    if not np.isfinite(array).all() or array.min() < -1e-6:
        raise ValueError("Projection must be finite nonnegative log attenuation")
    if expected_size and array.shape != (expected_size, expected_size):
        raise ValueError("Projection resolution differs from config; resample with calibrated pixel geometry")
    return torch.from_numpy(array.copy())[None]


class ProjectionDataset(Dataset):
    def __init__(self, manifest, split="train", image_size=256, projection_scale=1.0):
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        all_records = payload["scenes"]
        ids, groups = set(), {}
        for record in all_records:
            if record["id"] in ids:
                raise ValueError(f"Duplicate scene id: {record['id']}")
            ids.add(record["id"])
            group = record["group_id"]
            if group in groups and groups[group] != record["split"]:
                raise ValueError(f"Patient/object split leakage: {group}")
            groups[group] = record["split"]
            if record["split"] not in {"train", "val", "test"}:
                raise ValueError("Unknown split")
        self.records = [r for r in all_records if r["split"] == split]
        if not self.records:
            raise ValueError(f"No scenes in split {split}")
        self.split, self.image_size, self.projection_scale = split, image_size, projection_scale

    def __len__(self):
        return len(self.records)

    def read_scene(self, index):
        record = self.records[index]
        with np.load(self.root / record["projections"], allow_pickle=False) as archive:
            images = archive["projections"].astype(np.float32)
        if images.ndim == 4 and images.shape[1] == 1:
            images = images[:, 0]
        if images.ndim != 3 or images.shape[1:] != (self.image_size, self.image_size):
            raise ValueError(f"{record['id']}: expected [V,{self.image_size},{self.image_size}] projections")
        if not np.isfinite(images).all() or images.min() < -1e-6:
            raise ValueError(f"{record['id']}: invalid log attenuation")
        geometry = json.loads((self.root / record["geometry"]).read_text(encoding="utf-8"))
        cameras = [Camera(**x) for x in geometry["views"]]
        if len(cameras) != len(images):
            raise ValueError("Number of calibrated cameras differs from projection count")
        input_index = int(record["input_index"])
        targets = record.get("target_indices", [i for i in range(len(images)) if i != input_index])
        if input_index in targets or len(set(targets)) != len(targets) or not 0 <= input_index < len(images) or any(i < 0 or i >= len(images) for i in targets):
            raise ValueError("Invalid or overlapping input/target indices")
        if len(targets) < 2:
            raise ValueError("At least two non-input views required for denoising and held-out supervision")
        canonical = Camera(**geometry["canonical"])
        return record, torch.from_numpy(images.copy())[:, None] / self.projection_scale, cameras, input_index, targets, canonical

    def __getitem__(self, index):
        record, images, cameras, observed, targets, canonical = self.read_scene(index)
        a, b = random.sample(targets, 2) if self.split == "train" else targets[:2]
        return {"measured": images[observed], "clean": images[a], "novel": images[b],
                "condition_camera": cameras[observed], "target_camera": cameras[a],
                "novel_camera": cameras[b], "canonical_camera": canonical, "id": record["id"]}


def collate_samples(samples):
    return {key: torch.stack([x[key] for x in samples]) if isinstance(samples[0][key], torch.Tensor) else [x[key] for x in samples] for key in samples[0]}
