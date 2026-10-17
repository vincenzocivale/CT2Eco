"""Pre-training checks on a built dataset; exits non-zero if a hard check fails.

    python scripts/check_dataset.py data/pilot_v0 [--overlays]

1. split integrity   no CT case in two splits, every query belongs to a sweep of its split
2. round trip        source -> target -> source returns the source pixel
3. alignment         stored label slices agree with the 3D labels at query points
                     (source: must be exact; in-slice targets: reported, boundaries make it < 100%)
4. visibility audit  in-FOV / in-slice / out-of-plane rates per stratum
5. decorrelation     frame NCC vs out-of-plane displacement: does the image encode how far the plane moved?
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct2eco.geometry import apply, pixel_to_probe, probe_to_pixel  # noqa: E402
from ct2eco.queries import load_sweep, split_sweeps  # noqa: E402


DZ_BINS = [0, 0.2, 0.4, 0.8, 1.5, 3, 6, 12, 50]


def decorrelation(sweeps):
    """Mean NCC between frame pairs, binned by the mean |z| displacement of the image plane between them."""
    rows = []
    for sweep in sweeps:
        f, P, meta = sweep["frames"].astype(np.float32), sweep["poses"], sweep["meta"]
        H, W = meta["image_size"]
        v, u = np.mgrid[20:H:20, 10:W:20]
        pts = pixel_to_probe(np.stack([u.ravel(), v.ravel()], 1), meta["spacing_mm"], W)
        f = f - f.mean(axis=(1, 2), keepdims=True)
        f /= np.sqrt((f ** 2).sum(axis=(1, 2), keepdims=True)) + 1e-9
        for i in range(len(f)):
            for j in range(i + 1, len(f)):
                rows.append((np.abs(apply(np.linalg.inv(P[j]) @ P[i], pts)[:, 2]).mean(), (f[i] * f[j]).sum()))
    r = np.array(rows)
    print("\n[decorrelation] mean |dz| (mm) -> NCC")
    for lo, hi in zip(DZ_BINS[:-1], DZ_BINS[1:]):
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if m.any():
            print(f"  {lo:5.1f}-{hi:5.1f}  {r[m, 1].mean():.3f} +- {r[m, 1].std():.3f}  (n={m.sum()})")


def overlay(sweep, q, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    F = len(sweep["frames"])
    frames = [0, F // 2, F - 1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ax, f in zip(axes, frames):
        ax.imshow(sweep["frames"][f], cmap="gray")
        ax.contour(sweep["labels"][f], levels=np.unique(sweep["labels"][f]) + 0.5, colors="r", linewidths=0.4)
        rows = q["target_frame"] == f
        ax.scatter(q["target_u"][rows & q["target_in_slice"]], q["target_v"][rows & q["target_in_slice"]], s=6, c="lime")
        ax.set_title(f"{sweep['id']} frame {f}\nred: CT label boundaries, green: in-slice targets")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path)
    p.add_argument("--overlays", action="store_true", help="write checks/<sweep>.png for the first sweep of each split")
    a = p.parse_args(argv)
    root, ok = a.root, True

    splits = json.loads((root / "metadata" / "splits.json").read_text())
    seen = {}
    for split, ids in splits.items():
        for cid in ids:
            if cid in seen:
                print(f"FAIL split integrity: {cid} in {seen[cid]} and {split}")
                ok = False
            seen[cid] = split

    for split in splits:
        path = root / "queries" / f"{split}.parquet"
        if not path.exists():
            continue
        q = {k: np.asarray(v) for k, v in pq.read_table(path).to_pydict().items()}
        sweeps = {d.name: load_sweep(d) for d in split_sweeps(root, split)}
        if missing := set(q["sweep_id"]) - set(sweeps):
            print(f"FAIL split integrity: {split} queries reference sweeps outside the split: {sorted(missing)[:5]}")
            ok = False
            continue

        rt_err, src_agree, tgt_agree, n_tgt = 0.0, 0, 0, 0
        for sid, sweep in sweeps.items():
            r = q["sweep_id"] == sid
            if not r.any():
                continue
            meta, poses = sweep["meta"], sweep["poses"]
            W = meta["image_size"][1]
            src, tgt = q["source_frame"][r], q["target_frame"][r]
            p_t = np.stack([q["target_x_mm"][r], q["target_y_mm"][r], q["target_z_mm"][r]], 1)
            back = apply(np.linalg.inv(poses[src]) @ poses[tgt], p_t)
            uv_back = probe_to_pixel(back, meta["spacing_mm"], W)
            rt_err = max(rt_err, np.abs(uv_back - np.stack([q["source_u"][r], q["source_v"][r]], 1)).max(), np.abs(back[:, 2]).max())

            label = q["anatomical_label"][r]
            src_agree += (sweep["labels"][src, q["source_v"][r], q["source_u"][r]] == label).sum()
            s = q["target_in_slice"][r]
            ut = np.clip(np.rint(q["target_u"][r][s]).astype(int), 0, W - 1)
            vt = np.clip(np.rint(q["target_v"][r][s]).astype(int), 0, meta["image_size"][0] - 1)
            tgt_agree += (sweep["labels"][tgt[s], vt, ut] == label[s]).sum()
            n_tgt += s.sum()

            if a.overlays and sid == next(iter(sweeps)):
                (root / "checks").mkdir(exist_ok=True)
                overlay(sweep, {k: v[r] for k, v in q.items()}, root / "checks" / f"{sid}.png")

        n = len(q["sweep_id"])
        print(f"\n[{split}] {len(sweeps)} sweeps, {n} queries")
        print(f"  round trip max error      {rt_err:.2e} (px / mm)")
        print(f"  source label agreement    {src_agree / n:.4f}")
        print(f"  in-slice target agreement {tgt_agree / max(n_tgt, 1):.4f} over {n_tgt} queries")
        print(f"  outside CT volume         {(q['quality_flags'] & 1).mean():.4f}")
        for st in np.unique(q["stratum"]):
            m = q["stratum"] == st
            print(f"  {st:10s} {m.mean():.2f} | in_fov {q['target_in_fov'][m].mean():.2f} "
                  f"in_slice {q['target_in_slice'][m].mean():.2f} | median |z| {np.median(np.abs(q['target_z_mm'][m])):.1f} mm")
        if rt_err > 1e-6 or src_agree != n:
            print("  FAIL geometry: slicer and query labels disagree")
            ok = False

    decorrelation([load_sweep(d) for d in split_sweeps(root, "train")[:40]])
    print("\nOK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
