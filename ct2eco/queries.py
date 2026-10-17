"""Query sampling and labels (point correspondence + anatomy), and the torch Dataset that serves them."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy import ndimage

from .case import BACKGROUND, Case
from .geometry import apply, pixel_to_probe, probe_to_pixel

SDF_STRUCTURES = ["liver", "gallbladder", "kidney_right", "kidney_left", "spleen", "pancreas", "stomach", "aorta", "inferior_vena_cava"]
STRATA = {"interface": 0.5, "inside": 0.25, "other": 0.25}
NEAR_TARGET = 2  # half of the targets are within +-2 frames of the source, the rest uniform over the sweep
OUTSIDE_CT = 1  # quality_flags bit


def load_sweep(sweep_dir):
    d = Path(sweep_dir)
    return {
        "id": d.name,
        "frames": np.load(d / "frames.npy"),
        "labels": np.load(d / "labels.npy"),
        "poses": np.load(d / "poses.npy"),
        "meta": json.loads((d / "sweep.json").read_text()),
    }


def strata(labels):
    """(H, W) label slice -> stratum index per pixel (see STRATA order), -1 on air (not observable)."""
    edge = np.zeros(labels.shape, bool)
    for axis in (0, 1):
        diff = np.diff(labels, axis=axis) != 0
        lo = [slice(None)] * 2
        hi = [slice(None)] * 2
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        edge[tuple(lo)] |= diff
        edge[tuple(hi)] |= diff
    edge = ndimage.binary_dilation(edge, iterations=2)
    s = np.where(edge, 0, np.where(labels >= 2, 1, 2))
    s[labels == BACKGROUND] = -1
    return s


def sample_pixels(labels, n, rng):
    """n (u, v) pixels of one frame, stratified by STRATA; empty strata give their share to the others."""
    s = strata(labels).ravel()
    frac = np.array(list(STRATA.values()))
    counts = np.bincount(s[s >= 0], minlength=len(frac))
    if counts.sum() == 0:
        raise ValueError("frame has no tissue pixels")
    w = np.zeros(s.shape)
    ok = s >= 0
    w[ok] = (frac / np.maximum(counts, 1))[s[ok]]
    idx = rng.choice(s.size, size=n, p=w / w.sum())
    v, u = np.unravel_index(idx, labels.shape)
    return np.stack([u, v], 1), s[idx]


def compute_labels(sweep, case, src, tgt, uv):
    """Ground truth for queries (u, v, t_s, t_t): the source pixel expressed in the target probe frame."""
    meta = sweep["meta"]
    (H, W), spacing = meta["image_size"], meta["spacing_mm"]
    poses = sweep["poses"]
    p_world = apply(poses[src], pixel_to_probe(uv, spacing, W))
    p_t = apply(np.linalg.inv(poses[tgt]), p_world)
    uv_t = probe_to_pixel(p_t, spacing, W)
    in_fov = (uv_t[:, 0] >= -0.5) & (uv_t[:, 0] <= W - 0.5) & (uv_t[:, 1] >= -0.5) & (uv_t[:, 1] <= H - 0.5)
    return {
        "target_x_mm": p_t[:, 0], "target_y_mm": p_t[:, 1],
        "target_z_mm": p_t[:, 2],  # signed distance from the target plane, + along the probe elevation axis
        "target_u": uv_t[:, 0], "target_v": uv_t[:, 1],
        "target_in_fov": in_fov,  # orthogonal projection falls inside the image rectangle
        "target_in_slice": in_fov & (np.abs(p_t[:, 2]) <= meta["slice_thickness_mm"] / 2),
        "anatomical_label": case.labels_at(p_world),
        "sdf_mm": case.sdf_at(p_world, SDF_STRUCTURES),
        "quality_flags": np.where(case.inside_volume(p_world), 0, OUTSIDE_CT).astype(np.uint8),
    }


def sample_queries(sweep, case, n, rng):
    F = len(sweep["frames"])
    src = np.sort(rng.integers(F, size=n))
    uv = np.zeros((n, 2), np.int64)
    stratum = np.zeros(n, np.int64)
    for f in np.unique(src):
        rows = np.flatnonzero(src == f)
        uv[rows], stratum[rows] = sample_pixels(sweep["labels"][f], len(rows), rng)
    near = np.clip(src + rng.integers(-NEAR_TARGET, NEAR_TARGET + 1, n), 0, F - 1)
    tgt = np.where(rng.random(n) < 0.5, near, rng.integers(F, size=n))
    return {
        "sweep_id": np.full(n, sweep["id"]),
        "source_frame": src, "target_frame": tgt,
        "source_u": uv[:, 0], "source_v": uv[:, 1],
        "stratum": np.array(list(STRATA))[stratum],
        **compute_labels(sweep, case, src, tgt, uv),
    }


def to_table_dict(q):
    """Columns for pyarrow: sdf_mm (N, K) becomes a list column."""
    return {k: (list(v) if k == "sdf_mm" else v) for k, v in q.items()}


def split_sweeps(root, split):
    root = Path(root)
    cases = set(json.loads((root / "metadata" / "splits.json").read_text())[split])
    return [d for d in sorted((root / "sweeps").iterdir())
            if json.loads((d / "sweep.json").read_text())["case_id"] in cases]


class QueryDataset(torch.utils.data.Dataset):
    """One item = one sweep video + its queries. fixed=True reads queries/<split>.parquet, else samples online."""

    def __init__(self, root, split, queries_per_video=256, fixed=False, clip_len=None, sweeps=None):
        """clip_len: online -> a random window of that many frames; fixed -> the first clip_len frames and the
        queries inside them (variable count per item, use batch_size=1). sweeps: optional subset of sweep ids."""
        self.root = Path(root)
        self.sweeps = [d for d in split_sweeps(root, split) if sweeps is None or d.name in sweeps]
        if not self.sweeps:
            raise ValueError(f"no sweeps selected in split '{split}'")
        self.n = queries_per_video
        self.clip_len = clip_len
        self.fixed = None
        if fixed:
            t = pq.read_table(self.root / "queries" / f"{split}.parquet").to_pydict()
            ids = np.array(t["sweep_id"])
            self.fixed = {s.name: {k: np.asarray(v)[ids == s.name] for k, v in t.items() if k != "sweep_id"} for s in self.sweeps}
        self._cases = {}

    def __len__(self):
        return len(self.sweeps)

    def _case(self, case_id):
        if case_id not in self._cases:
            self._cases[case_id] = Case(self.root / "cases" / case_id, self.root / "metadata" / "label_map.json")
        return self._cases[case_id]

    def __getitem__(self, i):
        sweep = load_sweep(self.sweeps[i])
        if self.fixed is not None:
            q = self.fixed[sweep["id"]]
            if self.clip_len:
                sweep.update({k: sweep[k][:self.clip_len] for k in ("frames", "labels", "poses")})
                keep = (q["source_frame"] < self.clip_len) & (q["target_frame"] < self.clip_len)
                q = {k: v[keep] for k, v in q.items()}
        else:
            rng = np.random.default_rng()  # OS entropy: DataLoader workers do not reseed numpy's global RNG
            if self.clip_len:
                start = rng.integers(len(sweep["frames"]) - self.clip_len + 1)
                sweep.update({k: sweep[k][start:start + self.clip_len] for k in ("frames", "labels", "poses")})
            q = sample_queries(sweep, self._case(sweep["meta"]["case_id"]), self.n, rng)
            q = {k: v for k, v in q.items() if k not in ("sweep_id", "stratum")}
        q = {k: torch.as_tensor(np.asarray(v, np.float32 if np.asarray(v).dtype.kind == "f" else np.int64))
             for k, v in q.items() if k != "stratum"}
        return {
            "sweep_id": sweep["id"],
            "frames": torch.from_numpy(sweep["frames"]).float().div(255)[:, None],  # (F, 1, H, W): the only model input
            "poses": torch.from_numpy(sweep["poses"]).float(),  # reference only, not a model input
            "spacing_mm": torch.tensor(sweep["meta"]["spacing_mm"], dtype=torch.float32),  # (dy, dx) calibration
            "queries": q,
        }
