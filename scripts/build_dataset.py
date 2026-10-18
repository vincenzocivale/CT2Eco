"""Build the pilot dataset: TotalSegmentator subjects -> prepared cases -> rendered sweeps -> fixed queries.

    python scripts/build_dataset.py --totalseg-dir /data/Totalsegmentator_dataset_v201 --out data/pilot_v0
"""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct2eco.case import BACKGROUND, Case, build_label_map, prepare_case, structure_names  # noqa: E402
from ct2eco.queries import load_sweep, sample_queries, split_sweeps, to_table_dict  # noqa: E402
from ct2eco.render import render_sweep  # noqa: E402
from ct2eco.sweep import TARGETS, make_trajectory  # noqa: E402

MAX_AIR = 0.5  # reject sweeps whose frames are on average more than half air (probe off the body)
MAX_FRAME_AIR = 0.9  # pure-motion sweeps keep a fixed skin anchor and can drift off the body: no near-empty frame allowed


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--totalseg-dir", type=Path, required=True, help="folder of TotalSegmentator subjects (ct.nii.gz + segmentations/)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cases", type=int, default=20)
    p.add_argument("--sweeps", type=int, default=8, help="sweeps per case")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--queries", type=int, default=256, help="fixed queries per sweep written to queries/<split>.parquet")
    p.add_argument("--image-size", type=int, nargs=2, default=[200, 150], metavar=("H", "W"), help="SonoGym default")
    p.add_argument("--spacing-mm", type=float, default=0.4, help="isotropic pixel size; 0.4 mm is SonoGym's model-based setting")
    p.add_argument("--slice-thickness-mm", type=float, default=1.0, help="geometric slice thickness used by target_in_slice")
    p.add_argument("--require", default="liver", help="only use subjects where this structure is segmented")
    p.add_argument("--require-whole", action="store_true", help="... and not cut by any face of the CT volume")
    p.add_argument("--require-min-ml", type=float, default=0, help="... and at least this volume")
    p.add_argument("--motion", default="random", choices=["random", "lateral", "elevational",
                                                       "elevational_pure", "rot_x", "rot_y", "rot_z"], help="see sweep.make_trajectory")
    p.add_argument("--targets", nargs="+", default=TARGETS, help="organs the sweeps are centred on")
    p.add_argument("--no-ct", action="store_true", help="impedance from labels only (SonoGym if_use_ct=False)")
    p.add_argument("--beam-fwhm-mm", type=float, default=0.0, help="elevational beam width; 0 = thin plane (SonoGym)")
    p.add_argument("--beam-slices", type=int, default=7, help="sub-slices integrated across the beam")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def usable(mask_path, whole, min_ml):
    if not mask_path.exists():
        return False
    img = nib.load(mask_path)
    m = np.asarray(img.dataobj) > 0
    if m.sum() * abs(np.linalg.det(img.affine[:3, :3])) / 1000 < max(min_ml, 1e-9):
        return False
    faces = [np.take(m, i, axis=ax) for ax in range(3) for i in (0, -1)]
    return not (whole and any(f.any() for f in faces))


def pick_subjects(root, n, require, rng, whole=False, min_ml=0.0):
    subjects = [d for d in sorted(Path(root).iterdir()) if (d / "ct.nii.gz").exists()]
    picked = []
    for d in rng.permutation(subjects):
        if usable(d / "segmentations" / f"{require}.nii.gz", whole, min_ml):
            picked.append(d)
            if len(picked) == n:
                break
    if len(picked) < n:
        raise SystemExit(f"only {len(picked)} subjects with a usable '{require}' mask, asked for {n}")
    return sorted(picked)


def make_splits(case_ids, rng):
    """Case-level split (sweeps of one CT never cross splits): ~70/15/15 with at least one val and test case."""
    ids = list(rng.permutation(case_ids))
    k = max(1, round(0.15 * len(ids))) if len(ids) >= 3 else 0
    return {"train": sorted(ids[2 * k:]), "val": sorted(ids[:k]), "test": sorted(ids[k:2 * k])}


def main(argv=None):
    a = parse(argv)
    rng = np.random.default_rng(a.seed)
    out = a.out
    (out / "metadata").mkdir(parents=True, exist_ok=True)

    subjects = pick_subjects(a.totalseg_dir, a.cases, a.require, rng, a.require_whole, a.require_min_ml)
    lm_path = out / "metadata" / "label_map.json"
    names = sorted({n for s in subjects for n in structure_names(s)})
    if lm_path.exists():
        label_map = json.loads(lm_path.read_text())
    else:
        label_map = build_label_map(names)
        lm_path.write_text(json.dumps(label_map, indent=1))

    for ci, subject in enumerate(subjects):
        case_dir = out / "cases" / subject.name
        if not (case_dir / "segmentation.nii.gz").exists():
            prepare_case(subject, case_dir, label_map)
        case = Case(case_dir, lm_path)
        print(f"[{ci + 1}/{len(subjects)}] {case.id}", flush=True)
        for k in range(a.sweeps):
            sweep_dir = out / "sweeps" / f"{case.id}_sweep{k:02d}"
            if (sweep_dir / "sweep.json").exists():
                continue
            for attempt in range(50):
                seed = int(a.seed * 10**6 + ci * 10**4 + k * 100 + attempt)
                poses, view = make_trajectory(case, np.random.default_rng(seed), a.frames, a.targets, a.motion)
                if poses is None:
                    continue
                frames, labels = render_sweep(case, poses, a.image_size, [a.spacing_mm] * 2, seed, not a.no_ct, a.device,
                                              a.beam_fwhm_mm, a.beam_slices)
                air = labels == BACKGROUND
                if air.mean() <= MAX_AIR and (a.motion in ("random", "lateral", "elevational")
                                              or air.reshape(len(air), -1).mean(1).max() <= MAX_FRAME_AIR):
                    break
            else:
                raise RuntimeError(f"{case.id}: no valid sweep after 50 attempts")
            sweep_dir.mkdir(parents=True)
            np.save(sweep_dir / "frames.npy", frames)
            np.save(sweep_dir / "labels.npy", labels)
            np.save(sweep_dir / "poses.npy", poses)
            # written last: its presence marks a complete sweep, so an interrupted build resumes cleanly
            (sweep_dir / "sweep.json").write_text(json.dumps({
                "case_id": case.id, "seed": seed, "view": view, "probe": "linear",
                "image_size": a.image_size, "spacing_mm": [a.spacing_mm] * 2,
                "slice_thickness_mm": a.slice_thickness_mm, "poses": "probe->world (CT RAS mm), (F, 4, 4)",
                "simulator": "sonogym_conv", "use_ct": not a.no_ct, "targets": a.targets, "motion": a.motion,
                "beam_fwhm_mm": a.beam_fwhm_mm, "beam_slices": a.beam_slices,
            }, indent=1))

    splits_path = out / "metadata" / "splits.json"
    if not splits_path.exists():
        splits_path.write_text(json.dumps(make_splits([s.name for s in subjects], rng), indent=1))

    (out / "queries").mkdir(exist_ok=True)
    cases = {}
    for split in ("train", "val", "test"):
        parts = []
        for d in split_sweeps(out, split):
            sweep = load_sweep(d)
            cid = sweep["meta"]["case_id"]
            cases.setdefault(cid, Case(out / "cases" / cid, lm_path))
            parts.append(sample_queries(sweep, cases[cid], a.queries, np.random.default_rng(sweep["meta"]["seed"])))
        if parts:
            table = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            pq.write_table(pa.table(to_table_dict(table)), out / "queries" / f"{split}.parquet")
            print(f"{split}: {len(parts)} sweeps, {len(table['sweep_id'])} queries")


if __name__ == "__main__":
    main()
