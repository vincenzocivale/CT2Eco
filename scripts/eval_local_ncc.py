"""Image-domain NCC baseline for pure out-of-plane SonoGym motions.

Examples:
  python scripts/eval_local_ncc.py --data data/local_elev_pure --out runs/local/ncc_elev
  python scripts/eval_local_ncc.py --data data/local_rot_x --out runs/local/ncc_rx

No model training, no pose leakage: NCC maps use B-mode ONLY; pose GT is
used only to calibrate with TRAIN CTs, then evaluate on VAL CTs. No test set
is accessed. 1/2/4 frame-gap diagnostics are calibrated separately.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct2eco.local_ncc import spatial_ncc, dense_abs_z_from_poses
from ct2eco.local_ncc_baseline import fit_decreasing_curve, predict_curve, evaluate_dense


PURE_MOTIONS = {'elevational_pure', 'rot_x', 'rot_y'}


def collect(root, split, gap, pairs_per_sweep, device, windows):
    splits = json.loads((root / 'metadata' / 'splits.json').read_text())
    cases = set(splits[split])
    out = {'ncc': [], 'valid': [], 'abs_dz': []}
    sweep_count = 0
    for folder in sorted((root / 'sweeps').iterdir()):
        meta_path = folder / 'sweep.json'
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta['case_id'] not in cases:
            continue
        if meta.get('motion') not in PURE_MOTIONS:
            raise ValueError(f'Expected pure out-of-plane sweep, found {folder}: {meta.get("motion")}')
        frames = np.load(folder / 'frames.npy', mmap_mode='r')
        poses = np.load(folder / 'poses.npy', mmap_mode='r')
        if len(frames) <= gap:
            continue
        starts = np.unique(np.linspace(0, len(frames) - 1 - gap,
                                       min(pairs_per_sweep, len(frames) - gap)).astype(int))
        vids = torch.from_numpy(np.stack([np.stack([frames[i], frames[i + gap]])
                                          for i in starts]).copy()).to(device).float()[:, :, None] / 255.
        ps = torch.from_numpy(np.stack([np.stack([poses[i], poses[i + gap]])
                                        for i in starts]).copy()).to(device).float()
        spacings = torch.tensor(meta['spacing_mm'], device=device, dtype=ps.dtype).expand(len(starts), 2)
        with torch.no_grad():
            ncc, valid = spatial_ncc(vids, windows)
            target = dense_abs_z_from_poses(ps, vids.shape[-2:], spacings, ncc.shape[-2:])
        out['ncc'].append(ncc[:, 0].cpu().numpy())
        out['valid'].append(valid[:, 0].cpu().numpy())
        out['abs_dz'].append(target[:, 0, 0].cpu().numpy())
        sweep_count += 1
    if sweep_count == 0:
        raise ValueError(f'No sweeps in {split} at {root}')
    return {k: np.concatenate(v) for k, v in out.items()}, sweep_count


def fit_and_score(train, val, windows):
    valid_tr = train['valid'].all(axis=1)
    valid_val = val['valid'].all(axis=1)
    prior = np.zeros(train['abs_dz'].shape[1:], dtype=np.float32)
    global_median = np.median(train['abs_dz'][valid_tr])
    for i in range(prior.shape[0]):
        for j in range(prior.shape[1]):
            values = train['abs_dz'][:, i, j][valid_tr[:, i, j]]
            prior[i, j] = np.median(values) if len(values) else global_median
    curves = []
    vals = []
    for idx, window in enumerate(windows):
        curve = fit_decreasing_curve(train['ncc'][:, idx][valid_tr], train['abs_dz'][valid_tr])
        curves.append({'window': window, 'ncc': curve[0], 'abs_dz_mm': curve[1]})
        vals.append(predict_curve(val['ncc'][:, idx], curve))
    estimate = np.mean(vals, axis=0)
    return evaluate_dense(estimate, val['abs_dz'], valid_val, prior), curves


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gaps', type=int, nargs='+', default=[1, 2, 4])
    p.add_argument('--windows', type=int, nargs='+', default=[15, 31])
    p.add_argument('--pairs-per-sweep', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if a.pairs_per_sweep < 1 or any(g < 1 for g in a.gaps):
        p.error('pairs-per-sweep and frame gaps must be positive')
    a.out.mkdir(parents=True, exist_ok=True)
    report = {'data': str(a.data), 'windows': a.windows, 'gaps': {}}
    for gap in a.gaps:
        tr, ntr = collect(a.data, 'train', gap, a.pairs_per_sweep, a.device, a.windows)
        va, nva = collect(a.data, 'val', gap, a.pairs_per_sweep, a.device, a.windows)
        score, curves = fit_and_score(tr, va, a.windows)
        report['gaps'][str(gap)] = {'train_sweeps': ntr, 'val_sweeps': nva, **score,
                                    'calibrators_train_only': curves}
        print(f'gap={gap}: NCC MAE {score["ncc_mae_mm"]:.4f} mm | '
              f'spatial prior {score["spatial_prior_mae_mm"]:.4f} mm | '
              f'ratio {score["ratio_vs_spatial_prior"]:.3f} | coverage {score["coverage"]:.2%}', flush=True)
    (a.out / 'ncc_val.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Calibration: TRAIN only. Evaluation: VAL only. TEST untouched.', flush=True)


if __name__ == '__main__':
    main()
