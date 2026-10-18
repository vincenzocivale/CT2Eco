"""Sweep-level drift of frozen L1 motion, in the spirit of TUS-REC2024 metrics.

Per adjacent transition, the dense (dx, dy, |dz|) prediction is turned into a
rigid transform by a weighted Kabsch fit over the source grid, the transforms
are chained over the whole sweep, and image-grid points are compared with GT
in world coordinates (mm):
  GPE   mean point error over all frames of a sweep (global),
  FDE   point error at the last frame (final drift),
  LPE   mean point error of each single transition (local, frame-to-frame).

The elevation SIGN is not predicted by L1. Variants:
  oracle_sign  sign of dz taken from GT (diagnostic upper bound, NOT image-only),
  inplane      dz forced to 0 (image-only, ignores elevation),
  triangle     image-only RELATIVE sign (Housden-style 3-frame test on L1 |dz|
               at gaps 1 and 2) + ONE GT sign bit per sweep (first transition),
  identity     probe assumed static.
VAL only; test untouched.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct2eco.local_l1_alignment import load_l1_dense_checkpoint


def grid_points(image_hw, feature_hw, spacing):
    h, w = image_hw
    fh, fw = feature_hw
    y = (torch.arange(fh, dtype=torch.float64) * 4).clamp(max=h - 1) * spacing[0]
    x = ((torch.arange(fw, dtype=torch.float64) * 4).clamp(max=w - 1) - (w - 1) / 2) * spacing[1]
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((xx, yy, torch.zeros_like(xx)), -1).reshape(-1, 3)


def kabsch(src, dst):
    """Rigid 4x4 mapping src -> dst (least squares)."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = torch.linalg.svd((src - cs).T @ (dst - cd))
    d = torch.sign(torch.linalg.det(vt.T @ u.T))
    r = vt.T @ torch.diag(torch.tensor([1., 1., float(d)], dtype=src.dtype)) @ u.T
    out = torch.eye(4, dtype=src.dtype)
    out[:3, :3], out[:3, 3] = r, cd - r @ cs
    return out


@torch.no_grad()
def predict_transitions(encoder, head, frames, mode):
    """[F-1,3,h,w] mm. mode='full': one pass; 'win5': centred 5-frame windows."""
    if mode == 'full':
        return head(encoder(frames)['dense'])[0].double().cpu()
    n = frames.shape[1]
    out = []
    for t in range(n - 1):
        s = min(max(t - 1, 0), n - 5)
        out.append(head(encoder(frames[:, s:s + 5])['dense'])[0, t - s])
    return torch.stack(out).double().cpu()


@torch.no_grad()
def predict_gap2_abs_dz(encoder, head, frames):
    """[F-2,N] |dz| between frames t and t+2, from L1 run on stride-2 clips."""
    out = torch.zeros(frames.shape[1] - 2, *head(encoder(frames[:, :2])['dense']).shape[-2:],
                      dtype=torch.float64)
    for offset in (0, 1):
        dz = head(encoder(frames[:, offset::2])['dense'])[0, :, 2].double().cpu()
        out[offset::2] = dz[:len(out[offset::2])]
    return out.flatten(1)


def triangle_signs(abs1, abs2):
    """abs1 [F-1,N], abs2 [F-2,N] -> [F-1,N] signs relative to transition 0.

    Same direction if |d(t,t+2)| is closer to a+b than to |a-b|.
    """
    a, b = abs1[:-1], abs1[1:]
    flip = (abs2 - (a - b).abs()).abs() < (abs2 - (a + b)).abs()
    rel = torch.where(flip, -1., 1.).to(abs1.dtype)
    return torch.cat([torch.ones_like(abs1[:1]), torch.cumprod(rel, 0)])


def sweep_errors(pred, poses, points, abs2=None):
    """pred [F-1,3,h,w]; poses [F,4,4] float64; points [N,3] -> dict of errors."""
    hom = torch.cat([points, torch.ones(len(points), 1, dtype=points.dtype)], 1)
    variants = {}
    names = ['oracle_sign', 'inplane', 'identity']
    if abs2 is not None:
        signs = triangle_signs(pred[:, 2].flatten(1), abs2)
        gt_z = torch.stack([(hom @ (torch.linalg.inv(poses[t + 1]) @ poses[t]).T)[:, 2]
                            for t in range(len(poses) - 1)])
        bit = torch.sign(gt_z[0].mean())  # the single GT bit per sweep
        signs = signs * bit
        names.append('triangle')
        variants['triangle_sign_acc'] = float((signs == torch.sign(gt_z)).double().mean())
    for name in names:
        est = [poses[0]]
        local = []
        for t in range(len(poses) - 1):
            rel_gt = torch.linalg.inv(poses[t + 1]) @ poses[t]
            if name == 'identity':
                rel = torch.eye(4, dtype=poses.dtype)
            else:
                d = pred[t].reshape(3, -1).T
                dz = d[:, 2]
                if name == 'oracle_sign':
                    dz = dz * torch.sign((hom @ rel_gt.T)[:, 2])
                elif name == 'triangle':
                    dz = dz * signs[t]
                else:
                    dz = torch.zeros_like(dz)
                dst = points + torch.stack((d[:, 0], d[:, 1], dz), 1)
                rel = kabsch(points, dst)
            local.append(float(((hom @ rel.T)[:, :3] - (hom @ rel_gt.T)[:, :3]).norm(dim=1).mean()))
            est.append(est[-1] @ torch.linalg.inv(rel))
        errs = [float(((hom @ e.T)[:, :3] - (hom @ g.T)[:, :3]).norm(dim=1).mean())
                for e, g in zip(est, poses)]
        variants[name] = {'gpe_mm': float(np.mean(errs)), 'fde_mm': errs[-1],
                          'lpe_mm': float(np.mean(local))}
    return variants


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--l1-checkpoint', type=Path, required=True)
    p.add_argument('--mode', choices=['full', 'win5'], default='full')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    device = torch.device(a.device)
    encoder, head = load_l1_dense_checkpoint(a.l1_checkpoint, device)
    val = set(json.loads((a.data / 'metadata/splits.json').read_text())['val'])
    rows = []
    for sw in sorted((a.data / 'sweeps').iterdir()):
        if not (sw / 'sweep.json').exists():
            continue
        meta = json.loads((sw / 'sweep.json').read_text())
        if meta['case_id'] not in val:
            continue
        frames = torch.from_numpy(np.load(sw / 'frames.npy').astype(np.float32))[None, :, None].to(device) / 255
        poses = torch.from_numpy(np.load(sw / 'poses.npy').astype(np.float64))
        pred = predict_transitions(encoder, head, frames, a.mode)
        points = grid_points(frames.shape[-2:], pred.shape[-2:], meta['spacing_mm'])
        path_mm = float(sum((poses[t + 1, :3, 3] - poses[t, :3, 3]).norm() for t in range(len(poses) - 1)))
        abs2 = predict_gap2_abs_dz(encoder, head, frames)
        rows.append({'sweep': sw.name, 'case': meta['case_id'], 'path_mm': path_mm,
                     **sweep_errors(pred, poses, points, abs2)})
    summary = {'triangle_sign_acc': float(np.mean([r['triangle_sign_acc'] for r in rows]))}
    for name in ('oracle_sign', 'inplane', 'triangle', 'identity'):
        summary[name] = {k: float(np.mean([r[name][k] for r in rows])) for k in ('gpe_mm', 'fde_mm', 'lpe_mm')}
        summary[name]['fde_pct_path'] = float(np.mean([100 * r[name]['fde_mm'] / r['path_mm'] for r in rows]))
        summary[name]['by_ct_gpe_mm'] = {c: float(np.mean([r[name]['gpe_mm'] for r in rows if r['case'] == c]))
                                         for c in sorted({r['case'] for r in rows})}
    report = {'checkpoint': str(a.l1_checkpoint), 'mode': a.mode, 'split': 'val',
              'n_sweeps': len(rows), 'mean_path_mm': float(np.mean([r['path_mm'] for r in rows])),
              'summary': summary, 'sweeps': rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('mode', 'n_sweeps', 'mean_path_mm', 'summary')}, indent=2))


if __name__ == '__main__':
    main()
