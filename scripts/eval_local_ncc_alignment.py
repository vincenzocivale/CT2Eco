"""Matched diagnostic of fixed vs oracle vs image-estimated NCC on mixed SonoGym.

Only TRAIN cases fit monotone NCC->|dz| calibration and a spatial prior.
Only VAL cases are scored. No checkpoint training, test access or pose input
into the image-only alignment. Oracle is a *label-using diagnostic upper bound*,
not a deployable estimator; its projected registration need not be optimal.

Run from CT2Eco root:
  python scripts/eval_local_ncc_alignment.py --data data/pilot_v2 \
      --out runs/local/ncc_alignment_mixed --gaps 1 2 4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct2eco.local_ncc_alignment import three_way_ncc
from ct2eco.local_ncc_baseline import fit_decreasing_curve, predict_curve

MODES = ('fixed', 'oracle', 'estimated')
DZ_BINS = ((0., 0.5), (0.5, 1.5), (1.5, 3.), (3., float('inf')))
XY_BINS = ((0., 0.5), (0.5, 1.5), (1.5, 3.), (3., float('inf')))


def _sweep_paths(root, split):
    splits = json.loads((root / 'metadata' / 'splits.json').read_text())
    parts = {name: set(splits[name]) for name in ('train', 'val', 'test')}
    if any(parts[a] & parts[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('TRAIN/VAL/TEST CT case leakage detected in splits.json')
    cases = set(splits[split])
    paths = []
    for folder in sorted((root / 'sweeps').iterdir()):
        meta_path = folder / 'sweep.json'
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        if meta['case_id'] not in cases:
            continue
        # sweeps built before the 'motion' field existed were all random-motion
        if meta.get('motion', 'random') != 'random':
            raise ValueError(f'Expected motion=random in {folder}, found {meta["motion"]}')
        if 'spacing_mm' not in meta:
            raise ValueError(f'No spacing_mm in {folder}')
        paths.append((folder, meta))
    if not paths:
        raise ValueError(f'No mixed-motion sweeps for split {split} at {root}')
    return paths


@torch.no_grad()
def collect(root, split, gap, pairs_per_sweep, batch_size, device, windows,
            downsample, radius, patch):
    """Deterministic sweep-balanced sample; all modes share identical pixels."""
    arrays = {k: [] for k in (*MODES, 'valid', 'dz', 'xy_mm', 'flow_epe_px',
                              'estimated_confidence')}
    case_ids, sweep_ids = [], []
    paths = _sweep_paths(root, split)
    for folder, meta in paths:
        frames = np.load(folder / 'frames.npy', mmap_mode='r')
        poses = np.load(folder / 'poses.npy', mmap_mode='r')
        if len(frames) <= gap or len(frames) != len(poses):
            raise ValueError(f'Invalid frames/poses for gap={gap}: {folder}')
        starts = np.unique(np.linspace(0, len(frames) - gap - 1,
                                       min(pairs_per_sweep, len(frames) - gap)).astype(int))
        for j in range(0, len(starts), batch_size):
            ix = starts[j:j + batch_size]
            vid = np.stack([np.stack((frames[i], frames[i + gap])) for i in ix]).copy()
            ps = np.stack([np.stack((poses[i], poses[i + gap])) for i in ix]).copy()
            if vid.ndim != 4:
                raise ValueError(f'Unexpected image shape in {folder}: {vid.shape}')
            v = torch.from_numpy(vid).to(device).float()[:, :, None] / 255
            p = torch.from_numpy(ps).to(device).float()
            spacing = torch.tensor(meta['spacing_mm'], device=device, dtype=p.dtype).expand(len(ix), 2)
            out = three_way_ncc(v, p, spacing, windows=windows,
                                downsample=downsample, radius=radius, patch=patch)
            for mode in MODES:
                arrays[mode].append(out['ncc'][mode].cpu().numpy())
            for key in ('valid', 'dz', 'xy_mm', 'flow_epe_px', 'estimated_confidence'):
                arrays[key].append(out[key].cpu().numpy())
            case_ids.extend([str(meta['case_id'])] * len(ix))
            sweep_ids.extend([folder.name] * len(ix))
    data = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    data['case_ids'] = np.asarray(case_ids)
    data['sweep_ids'] = np.asarray(sweep_ids)
    data['n_sweeps'] = len(paths)
    return data


def spatial_prior(train, mask):
    dz = train['dz']
    counts = mask.sum(axis=0)
    total = (dz * mask).sum(axis=0)
    global_fallback = float(dz[mask].mean())
    return np.where(counts > 0, total / np.maximum(counts, 1), global_fallback)


def fit_train_calibration(train, max_dz, windows, max_pixels=250000, seed=17):
    mask = train['valid'] & (train['dz'] <= max_dz)
    if not mask.any():
        raise ValueError('No matched training pixels within requested elevation range')
    prior = spatial_prior(train, mask)
    indices = np.flatnonzero(mask.ravel())
    if len(indices) > max_pixels:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_pixels, replace=False))
    curves = {}
    dz_sample = train['dz'].ravel()[indices]
    for method in MODES:
        curves[method] = []
        for k, width in enumerate(windows):
            ncc_sample = train[method][:, k].ravel()[indices]
            curve = fit_decreasing_curve(ncc_sample, dz_sample, bins=80)
            curves[method].append({'window_px': width, 'ncc': curve[0], 'dz_mm': curve[1]})
    return prior, curves, {'training_matched_pixels': int(mask.sum()),
                           'calibration_pixels': len(indices)}


def _predictions(val, curves):
    values = {}
    for method in MODES:
        predictions = []
        for k, c in enumerate(curves[method]):
            predictions.append(predict_curve(val[method][:, k], (c['ncc'], c['dz_mm'])))
        values[method] = np.mean(predictions, axis=0)
    return values


def _score(pred, target, mask):
    return float(np.abs(pred - target)[mask].mean()) if mask.any() else None


def report_validation(val, prior, curves, max_dz, bootstraps=1000, seed=17):
    # Common-support invariant: all three modes use exactly the same cells.
    support = val['valid']
    core = support & (val['dz'] <= max_dz)
    if not core.any():
        raise ValueError('No matched validation pixels in observability range')
    prediction = _predictions(val, curves)
    prediction['spatial_prior'] = np.broadcast_to(prior, val['dz'].shape)
    methods = (*MODES, 'spatial_prior')
    score = lambda mask: {name: _score(prediction[name], val['dz'], mask) for name in methods}
    result = {'n_cases': len(set(val['case_ids'])), 'n_sweeps': val['n_sweeps'],
              'n_pairs': len(val['dz']), 'n_common_pixels': int(support.sum()),
              'n_primary_pixels': int(core.sum()),
              'coverage': float(support.mean()),
              'primary_max_abs_dz_mm': max_dz,
              'primary_mae_mm': score(core), 'full_common_support_mae_mm': score(support),
              'by_abs_dz_mm': {}, 'by_inplane_mm': {}, 'by_case': {},
              'flow_epe_px': float(val['flow_epe_px'][core].mean()),
              'estimated_match_coverage': float(val['estimated_confidence'][core].mean()),
              'mean_ncc_on_primary_support': {m: float(val[m].mean(axis=1)[core].mean())
                                              for m in MODES}}
    for name, feature, bins in (('by_abs_dz_mm', val['dz'], DZ_BINS),
                                ('by_inplane_mm', val['xy_mm'], XY_BINS)):
        for lo, hi in bins:
            mask = support & (feature >= lo) & (feature < hi)
            # bins diagnose matching and saturation over the full common support
            if mask.any():
                result[name][f'{lo:g}-{hi:g}' if np.isfinite(hi) else f'{lo:g}-inf'] = {
                    'n_pixels': int(mask.sum()), 'mae_mm': score(mask)}
    case_metrics = []
    for case in sorted(set(val['case_ids'])):
        idx = val['case_ids'] == case
        mask = core & idx[:, None, None]
        if not mask.any():
            continue
        cm = {'n_primary_pixels': int(mask.sum()), 'mae_mm': score(mask)}
        result['by_case'][case] = cm
        case_metrics.append(cm['mae_mm'])
    for method in methods:
        result.setdefault('macro_ct_mae_mm', {})[method] = float(np.mean([s[method] for s in case_metrics]))
    if case_metrics and bootstraps:
        rng = np.random.default_rng(seed)
        n = len(case_metrics)
        ci = {}
        for method in ('oracle', 'estimated'):
            differences = np.asarray([case[method] - case['fixed'] for case in case_metrics])
            draws = rng.integers(0, n, size=(bootstraps, n))
            means = differences[draws].mean(axis=1)
            ci[f'{method}_minus_fixed'] = {
                'mean_ct_delta_mm': float(differences.mean()),
                'bootstrap_95pct_ct_CI_mm': np.quantile(means, [0.025, 0.975]).tolist(),
                'n_ct': n}
        result['paired_ct_bootstrap'] = ci
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gaps', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--windows', type=int, nargs='+', default=[15, 31])
    parser.add_argument('--train-pairs-per-sweep', type=int, default=4)
    parser.add_argument('--val-pairs-per-sweep', type=int, default=8)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--downsample', type=int, default=4)
    parser.add_argument('--match-radius', type=int, default=4)
    parser.add_argument('--match-window', type=int, default=7)
    parser.add_argument('--max-dz-mm', type=float, default=3.)
    parser.add_argument('--calibration-pixels', type=int, default=250000)
    parser.add_argument('--bootstraps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if (args.batch < 1 or args.train_pairs_per_sweep < 1 or args.val_pairs_per_sweep < 1
        or not args.gaps or any(g < 1 for g in args.gaps)
        or not args.windows or any(k < 3 or k % 2 == 0 for k in args.windows)
        or args.downsample < 1 or args.match_radius < 0
        or args.match_window < 3 or args.match_window % 2 == 0
        or args.max_dz_mm <= 0 or args.calibration_pixels < 100
        or args.bootstraps < 0):
        parser.error('Invalid pairs/batch/gaps/windows/matching/max-dz/calibration/bootstraps')
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    report = {'dataset': str(args.data), 'split_protocol': 'TRAIN fit -> VAL evaluate; no TEST',
              'oracle_warning': 'Uses pose-derived 2D projection (not necessarily best image correspondence). Diagnostic only.',
              'windows': args.windows,
              'parameters': {'gaps': args.gaps, 'max_dz_mm': args.max_dz_mm,
                             'train_pairs_per_sweep': args.train_pairs_per_sweep,
                             'val_pairs_per_sweep': args.val_pairs_per_sweep,
                             'downsample': args.downsample, 'match_radius': args.match_radius,
                             'match_window': args.match_window}, 'gaps': {}}
    for gap in args.gaps:
        train = collect(args.data, 'train', gap, args.train_pairs_per_sweep,
                        args.batch, device, args.windows,
                        args.downsample, args.match_radius, args.match_window)
        val = collect(args.data, 'val', gap, args.val_pairs_per_sweep,
                      args.batch, device, args.windows,
                      args.downsample, args.match_radius, args.match_window)
        prior, curves, fitting = fit_train_calibration(train, args.max_dz_mm,
                                                      args.windows, args.calibration_pixels,
                                                      seed=args.seed)
        result = report_validation(val, prior, curves, args.max_dz_mm,
                                   bootstraps=args.bootstraps, seed=args.seed)
        report['gaps'][str(gap)] = {'fit': fitting, 'validation': result,
                                    'calibration_train_only': curves}
        m = result['primary_mae_mm']
        print(f'gap={gap}  fixed={m["fixed"]:.4f}  oracle={m["oracle"]:.4f}  '
              f'estimated={m["estimated"]:.4f}  prior={m["spatial_prior"]:.4f} mm '
              f'common coverage={result["coverage"]:.1%}  '
              f'flow EPE={result["flow_epe_px"]:.2f}px', flush=True)
        (args.out / 'alignment_val.json').write_text(json.dumps(report, indent=2) + '\n')
    print('TRAIN-only calibration; VAL-only report; TEST untouched. '
          'Oracle geometry is diagnostic, never an image-only prediction.', flush=True)


if __name__ == '__main__':
    main()
