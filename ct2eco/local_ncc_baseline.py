"""Train-only monotone NCC-to-|dz| calibration for controlled SonoGym sweeps.

This is a NON-LEARNED image representation and a simple calibrated baseline.
Calibrators and spatial priors are fitted exclusively on training CT cases.
"""

import numpy as np


def fit_decreasing_curve(x, y, bins=80):
    """Quantile-bin + weighted pool-adjacent-violators for decreasing isotonic fit.

    Returns knot positions and non-increasing predicted |dz| in mm; no sklearn.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        raise ValueError('No valid calibration samples')
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) == 1:
        return [float(edges[0])], [float(np.median(y))]
    ids = np.clip(np.searchsorted(edges[1:-1], x, side='right'), 0, len(edges) - 2)
    nbin = len(edges) - 1
    counts = np.bincount(ids, minlength=nbin)
    means_x = np.bincount(ids, weights=x, minlength=nbin) / np.maximum(counts, 1)
    means_y = np.bincount(ids, weights=y, minlength=nbin) / np.maximum(counts, 1)
    blocks = []
    for i in np.flatnonzero(counts):
        blocks.append([int(i), int(i), float(means_y[i]), float(counts[i])])
        while len(blocks) > 1 and blocks[-2][2] < blocks[-1][2]:
            b = blocks.pop()
            a = blocks.pop()
            weight = a[3] + b[3]
            blocks.append([a[0], b[1], (a[2] * a[3] + b[2] * b[3]) / weight, weight])
    fitted = np.zeros(nbin, dtype=np.float64)
    for start, end, mean, _ in blocks:
        fitted[start:end + 1] = mean
    keep = counts > 0
    return means_x[keep].tolist(), fitted[keep].clip(min=0).tolist()


def predict_curve(x, curve):
    xp, yp = curve
    return np.interp(x, xp, yp, left=yp[0], right=yp[-1]).astype(np.float32)


def evaluate_dense(pred, true, valid, prior):
    true, pred = np.asarray(true), np.asarray(pred)
    valid = np.asarray(valid, dtype=bool)
    prior = np.broadcast_to(prior, true.shape)
    if not valid.any():
        raise ValueError('No valid NCC pixels for evaluation')
    summary = {'n_valid_pixels': int(valid.sum()),
               'coverage': float(valid.mean()),
               'ncc_mae_mm': float(np.abs(pred - true)[valid].mean()),
               'spatial_prior_mae_mm': float(np.abs(prior - true)[valid].mean())}
    summary['ratio_vs_spatial_prior'] = summary['ncc_mae_mm'] / max(summary['spatial_prior_mae_mm'], 1e-10)
    summary['by_abs_dz_mm'] = {}
    for lo, hi in ((0, 0.5), (0.5, 1.5), (1.5, 3), (3, 6), (6, float('inf'))):
        selection = valid & (true >= lo) & (true < hi)
        label = f'{lo}-{hi if np.isfinite(hi) else "inf"}'
        if selection.any():
            summary['by_abs_dz_mm'][label] = {
                'n': int(selection.sum()),
                'ncc_mae_mm': float(np.abs(pred - true)[selection].mean()),
                'spatial_prior_mae_mm': float(np.abs(prior - true)[selection].mean())}
    return summary
