"""Four-way validation-only NCC diagnostic using a frozen L1 xyz checkpoint.

Compares fixed, oracle, image-only classical matching and frozen L1 flow on the
SAME pixel support. Isotonic NCC->|dz| mappings and priors are calibrated on
TRAIN CTs, never VAL or TEST. The L1 checkpoint is evaluated, not trained.

Only gap=1 is supported: L1 was trained on *adjacent* frame transitions.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct2eco.local_l1_alignment import load_l1_dense_checkpoint, l1_inplane_flow
from ct2eco.local_ncc_alignment import three_way_ncc, aligned_ncc, oracle_flow
from ct2eco.local_ncc_baseline import fit_decreasing_curve, predict_curve
from scripts.eval_local_ncc_alignment import _sweep_paths, spatial_prior, DZ_BINS, XY_BINS

MODES = ('fixed', 'oracle', 'estimated', 'l1')


@torch.no_grad()
def collect(root, split, pairs_per_sweep, batch_size, device, windows,
            encoder, probe, downsample=4, radius=4, patch=7):
    if pairs_per_sweep < 1 or batch_size < 1:
        raise ValueError('Invalid pair count / batch size')
    paths = _sweep_paths(root, split)  # verifies disjoint CT case splits and random motion
    arrays = {name: [] for name in (*MODES, 'valid', 'dz', 'xy_mm',
                                   'classical_flow_epe_px', 'l1_flow_epe_px')}
    case_ids, sweep_ids = [], []
    for folder, meta in paths:
        frames = np.load(folder / 'frames.npy', mmap_mode='r')
        poses = np.load(folder / 'poses.npy', mmap_mode='r')
        if len(frames) != len(poses) or len(frames) < 5:
            raise ValueError(f'Need >=5 frames with aligned poses: {folder}')
        # All methods use identical adjacent pairs and can access 5 frames for L1.
        starts = np.unique(np.linspace(0, len(frames) - 5,
                           min(pairs_per_sweep, len(frames) - 4)).astype(int))
        for offset in range(0, len(starts), batch_size):
            ix = starts[offset:offset + batch_size]
            clips = np.stack([frames[i:i+5] for i in ix]).copy()
            pose_pairs = np.stack([poses[i:i+2] for i in ix]).copy()
            clip = torch.from_numpy(clips).to(device=device, dtype=torch.float32)[:, :, None] / 255
            pp = torch.from_numpy(pose_pairs).to(device=device, dtype=torch.float32)
            spacing = torch.as_tensor(meta['spacing_mm'], device=device, dtype=pp.dtype).expand(len(ix), 2)
            pair = clip[:, :2]
            base = three_way_ncc(pair, pp, spacing, windows=windows,
                                 downsample=downsample, radius=radius, patch=patch)
            l1_flow = l1_inplane_flow(encoder, probe, clip, spacing)
            h, w = pair.shape[-2:]
            l1_ncc, l1_valid = aligned_ncc(pair[:, 0], pair[:, 1], l1_flow,
                                           windows=windows, out_hw=base['dz'].shape[-2:])
            common = base['valid'] & l1_valid.all(dim=1)
            oracle, _ = oracle_flow(pp, spacing, (h, w))
            l1_epe = torch.linalg.vector_norm(l1_flow - oracle, dim=1)
            l1_epe = F.interpolate(l1_epe[:, None], size=common.shape[-2:], mode='area')[:, 0]
            out = {**base['ncc'], 'l1': l1_ncc,
                   'valid': common,
                   'dz': base['dz'], 'xy_mm': base['xy_mm'],
                   'classical_flow_epe_px': base['flow_epe_px'],
                   'l1_flow_epe_px': l1_epe}
            for key in arrays:
                arrays[key].append(out[key].cpu().numpy())
            case_ids.extend([str(meta['case_id'])] * len(ix))
            sweep_ids.extend([folder.name] * len(ix))
    data = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    data['case_ids'], data['sweep_ids'] = np.array(case_ids), np.array(sweep_ids)
    data['n_sweeps'] = len(paths)
    return data


def fit_calibration(train, windows, max_dz_mm, max_pixels=250000, seed=17):
    mask = train['valid'] & (train['dz'] <= max_dz_mm)
    if not mask.any():
        raise ValueError('No train pixels in calibrated range')
    prior = spatial_prior(train, mask)
    ids = np.flatnonzero(mask.ravel())
    if len(ids) > max_pixels:
        ids = np.sort(np.random.default_rng(seed).choice(ids, max_pixels, replace=False))
    target = train['dz'].ravel()[ids]
    curves = {}
    for method in MODES:
        curves[method] = []
        for j, k in enumerate(windows):
            x, y = fit_decreasing_curve(train[method][:, j].ravel()[ids], target, bins=80)
            curves[method].append({'window_px': int(k), 'ncc': x, 'dz_mm': y})
    return prior, curves, {'train_matched_pixels': int(mask.sum()),
                           'train_calibration_pixels': len(ids)}


def evaluate(val, prior, curves, max_dz_mm, seed=17, bootstraps=1000):
    support = val['valid']
    mask = support & (val['dz'] <= max_dz_mm)
    if not mask.any():
        raise ValueError('No validation pixels in calibrated range')
    predictions = {method: np.mean([
        predict_curve(val[method][:, k], (curve['ncc'], curve['dz_mm']))
        for k, curve in enumerate(curves[method])], axis=0) for method in MODES}
    predictions['spatial_prior'] = np.broadcast_to(prior, val['dz'].shape)
    methods = (*MODES, 'spatial_prior')
    mae = lambda pred, s: float(np.abs(pred - val['dz'])[s].mean()) if s.any() else None
    score = lambda s: {method: mae(predictions[method], s) for method in methods}
    epe = lambda name, s: float(val[name][s].mean()) if s.any() else None
    report = {
        'n_ct': len(set(val['case_ids'])), 'n_sweeps': val['n_sweeps'],
        'n_pairs': len(val['dz']), 'n_common_pixels': int(support.sum()),
        'n_primary_pixels': int(mask.sum()), 'common_coverage': float(support.mean()),
        'max_abs_dz_mm': float(max_dz_mm),
        'primary_mae_mm': score(mask), 'full_common_mae_mm': score(support),
        'flow_epe_px': {'classical': epe('classical_flow_epe_px', mask),
                        'l1': epe('l1_flow_epe_px', mask)},
        'by_abs_dz_mm': {}, 'by_inplane_mm': {}, 'by_ct': {},
    }
    for key, attr, bins in (('by_abs_dz_mm', 'dz', DZ_BINS),
                            ('by_inplane_mm', 'xy_mm', XY_BINS)):
        for lo, hi in bins:
            sel = support & (val[attr] >= lo) & (val[attr] < hi)
            if sel.any():
                name = f'{lo:g}-{hi:g}' if np.isfinite(hi) else f'{lo:g}-inf'
                report[key][name] = {'n_pixels': int(sel.sum()), 'mae_mm': score(sel),
                                     'flow_epe_px': {
                                         'classical': epe('classical_flow_epe_px', sel),
                                         'l1': epe('l1_flow_epe_px', sel)}}
    paired = []
    for case in sorted(set(val['case_ids'])):
        sel = mask & (val['case_ids'] == case)[:, None, None]
        if sel.any():
            scores = score(sel)
            report['by_ct'][case] = {'n_pixels': int(sel.sum()), 'mae_mm': scores,
                                     'flow_epe_px': {
                                         'classical': epe('classical_flow_epe_px', sel),
                                         'l1': epe('l1_flow_epe_px', sel)}}
            paired.append(scores)
    if paired:
        report['macro_ct_mae_mm'] = {key: float(np.mean([v[key] for v in paired]))
                                      for key in methods}
        if bootstraps:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(paired), size=(bootstraps, len(paired)))
            report['paired_ct_bootstrap'] = {}
            for key in ('oracle', 'estimated', 'l1'):
                values = np.asarray([v[key] - v['fixed'] for v in paired])
                draws = values[idx].mean(axis=1)
                report['paired_ct_bootstrap'][f'{key}_minus_fixed'] = {
                    'n_ct': len(paired), 'mean_ct_delta_mm': float(values.mean()),
                    'bootstrap_95pct_ci_mm': np.quantile(draws, [0.025, 0.975]).tolist()}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--l1-checkpoint', type=Path, required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--gap', type=int, default=1)
    p.add_argument('--train-pairs-per-sweep', type=int, default=4)
    p.add_argument('--val-pairs-per-sweep', type=int, default=8)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--windows', type=int, nargs='+', default=[15, 31])
    p.add_argument('--downsample', type=int, default=4)
    p.add_argument('--match-radius', type=int, default=4)
    p.add_argument('--match-window', type=int, default=7)
    p.add_argument('--max-dz-mm', type=float, default=3.)
    p.add_argument('--calibration-pixels', type=int, default=250000)
    p.add_argument('--bootstraps', type=int, default=1000)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if a.gap != 1:
        p.error('Only --gap 1 is valid: L1 predicts adjacent transitions from five frames')
    if (a.train_pairs_per_sweep < 1 or a.val_pairs_per_sweep < 1 or a.batch < 1
            or any(k < 3 or not k % 2 for k in a.windows) or not a.windows
            or a.downsample < 1 or a.match_radius < 0
            or a.match_window < 3 or a.match_window % 2 == 0
            or a.max_dz_mm <= 0 or a.calibration_pixels < 100 or a.bootstraps < 0):
        p.error('Invalid run parameters')
    torch.manual_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    encoder, probe = load_l1_dense_checkpoint(a.l1_checkpoint, device)
    opts = dict(batch_size=a.batch, device=device, windows=a.windows,
                encoder=encoder, probe=probe, downsample=a.downsample,
                radius=a.match_radius, patch=a.match_window)
    tr = collect(a.data, 'train', pairs_per_sweep=a.train_pairs_per_sweep, **opts)
    va = collect(a.data, 'val', pairs_per_sweep=a.val_pairs_per_sweep, **opts)
    prior, curves, fit = fit_calibration(tr, a.windows, a.max_dz_mm,
                                         a.calibration_pixels, a.seed)
    score = evaluate(va, prior, curves, a.max_dz_mm,
                     seed=a.seed, bootstraps=a.bootstraps)
    result = {'dataset': str(a.data), 'checkpoint': str(a.l1_checkpoint),
              'protocol': 'FIT TRAIN / SCORE VAL; TEST untouched; no model training',
              'gap': 1, 'windows': a.windows, 'fit': fit,
              'evaluation': score, 'train_only_calibrations': curves,
              'warning': 'Oracle uses poses and is not a deployable bound; L1 is image-only.'}
    output = a.out / 'l1_alignment_val.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    m = score['primary_mae_mm']
    e = score['flow_epe_px']
    print('gap=1  ' + '  '.join(f'{k}={m[k]:.4f}' for k in MODES) +
          f'  prior={m["spatial_prior"]:.4f} mm  coverage={score["common_coverage"]:.1%}', flush=True)
    print(f'flow EPE classical={e["classical"]:.3f}px L1={e["l1"]:.3f}px; '
          f'n_CT={score["n_ct"]}  n_pairs={score["n_pairs"]}', flush=True)
    print(f'VAL report: {output}; TEST not accessed.', flush=True)


if __name__ == '__main__':
    main()
