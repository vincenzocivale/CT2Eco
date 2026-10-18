"""Zero-training frozen-feature query matching on independent validation CTs.

Evaluate 0->gap (gap=1,2,4) using five-frame CT2Eco sweeps:
  * trained L1 frame_dense cosine NN (frozen; no trained matching head)
  * random-init L1 frame_dense cosine NN (architecture control)
  * native B-mode patch NCC matching (image-only classical control)
  * identity (no-motion control)
  * trained L1 dense motion head composed through adjacent frame steps
    (strong but NOT independent supervision: the original dense motion task)

Only labels come from poses. No queries are selected using target poses.
No fit, training or test split access. Queries with small |dz| form an
exploratory PROXY-overlap group, not validated physical correspondences.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct2eco.local_l1_alignment import load_l1_dense_checkpoint
from ct2eco.local_stream import LocalStream
from ct2eco.local_transfer import (compose_l1_displacements, match_descriptors,
                                   project_queries, query_metrics, query_positions,
                                   raw_patch_descriptors)
from scripts.eval_local_ncc_alignment import _sweep_paths

MODES = ('identity', 'raw_patch', 'random_l1', 'trained_l1', 'trained_motion_head')
GAPS = (1, 2, 4)


def add_metrics(aggregate, per_batch):
    for band, data in per_batch.items():
        dst = aggregate.setdefault(band, {'n': 0})
        dst['n'] += data['n']
        if 'search_reachable' in data:
            dst['search_reachable'] = dst.get('search_reachable', 0) + data['search_reachable']
        for method in MODES:
            if method not in data:
                continue
            src = data[method]
            d = dst.setdefault(method, {'epe_sum_px': 0., 'within_4px': 0, 'within_8px': 0})
            for key in d:
                d[key] += src[key]


def finalize(agg):
    out = {}
    for band, vals in agg.items():
        n = vals['n']
        out[band] = {'n_points': n}
        if n and 'search_reachable' in vals:
            out[band]['search_reachable_fraction'] = vals['search_reachable'] / n
        if n:
            for method in MODES:
                if method not in vals:
                    continue
                src = vals[method]
                out[band][method] = {
                    'epe_px': src['epe_sum_px'] / n,
                    'recall_at_4px': src['within_4px'] / n,
                    'recall_at_8px': src['within_8px'] / n}
    return out


@torch.inference_mode()
def evaluate(root, encoder, probe, random_encoder, device='cpu',
             clips_per_sweep=8, batch=4, num_queries=64, radius=8,
             raw_patch=9, max_dz_mm=0.5):
    paths = _sweep_paths(Path(root), 'val')
    overall = {gap: {} for gap in GAPS}
    by_ct = {gap: {} for gap in GAPS}
    sample_counts = {gap: 0 for gap in GAPS}
    evaluated_cts = set()
    for folder, meta in paths:
        frames = np.load(folder / 'frames.npy', mmap_mode='r')
        poses = np.load(folder / 'poses.npy', mmap_mode='r')
        if len(frames) != len(poses) or len(frames) < 5:
            raise ValueError(f'Invalid sweep with <5 frames: {folder}')
        case = str(meta['case_id'])
        evaluated_cts.add(case)
        starts = np.unique(np.linspace(0, len(frames) - 5,
                                       min(clips_per_sweep, len(frames) - 4)).astype(int))
        for i in range(0, len(starts), batch):
            start = starts[i:i+batch]
            v = np.stack([np.array(frames[k:k+5]) for k in start])
            p = np.stack([np.array(poses[k:k+5]) for k in start])
            video = torch.from_numpy(v).float().to(device)[:, :, None] / 255
            pp = torch.from_numpy(p).float().to(device)
            spacing = torch.as_tensor(meta['spacing_mm'], device=device,
                                      dtype=torch.float32).expand(len(start), 2)
            trained = encoder(video)
            random = random_encoder(video)
            frames_trained = trained['frame_dense']
            frames_random = random['frame_dense']
            if frames_trained is None or frames_random is None:
                raise ValueError('L1 frame_dense unavailable')
            fh, fw = frames_trained.shape[-2:]
            queries = query_positions((fh, fw), num_queries, device=device)
            raw = raw_patch_descriptors(video, (fh, fw), patch=raw_patch)
            dense_xyz = probe(trained['dense'])
            source = queries.float()[None].expand(len(start), -1, -1) * 4
            for gap in GAPS:
                target, dz, in_fov = project_queries(pp, spacing, queries,
                                                      video.shape[-2:], 0, gap)
                estimates = {
                    'identity': source,
                    'raw_patch': match_descriptors(raw[:, 0], raw[:, gap], queries,
                                                   radius=radius, centered=True)[0],
                    'random_l1': match_descriptors(frames_random[:, 0],
                                                   frames_random[:, gap], queries,
                                                   radius=radius)[0],
                    'trained_l1': match_descriptors(frames_trained[:, 0],
                                                    frames_trained[:, gap], queries,
                                                    radius=radius)[0],
                    'trained_motion_head': compose_l1_displacements(
                        dense_xyz, queries, spacing, gap),
                }
                per = query_metrics(target, dz, in_fov, estimates, max_dz_mm=max_dz_mm,
                                    search_radius=radius)
                add_metrics(overall[gap], per)
                add_metrics(by_ct[gap].setdefault(case, {}), per)
                sample_counts[gap] += len(start)
    summary = {}
    for gap in GAPS:
        full = finalize(overall[gap])
        cases = {case: finalize(values) for case, values in by_ct[gap].items()}
        macro = {}
        for band in ('proxy_overlap', 'fov_all'):
            for method in MODES:
                measured = [v[band][method]['epe_px'] for v in cases.values()
                            if band in v and method in v[band]]
                macro.setdefault(band, {})[method] = float(np.mean(measured)) if measured else None
        summary[str(gap)] = {'n_clips': sample_counts[gap], 'micro': full,
                             'macro_ct_epe_px': macro, 'by_ct': cases}
    return {'n_ct': len(evaluated_cts), 'case_ids': sorted(evaluated_cts),
            'n_sweeps': len(paths), 'gaps': summary}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--l1-checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--clips-per-sweep', type=int, default=8)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--queries', type=int, default=64)
    p.add_argument('--radius', type=int, default=8, help='Search radius in 1/4-resolution feature cells')
    p.add_argument('--raw-patch', type=int, default=9, help='Native-pixel B-mode patch width')
    p.add_argument('--proxy-max-dz-mm', type=float, default=0.5)
    p.add_argument('--random-seed', type=int, default=314159)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if min(a.clips_per_sweep, a.batch, a.queries) < 1 or a.radius < 0 or a.proxy_max_dz_mm <= 0:
        p.error('Invalid counts, radius or overlap proxy')
    if a.raw_patch < 3 or a.raw_patch % 2 != 1:
        p.error('--raw-patch must be odd and >=3')
    if not a.l1_checkpoint.exists():
        p.error(f'Checkpoint not found: {a.l1_checkpoint}')
    torch.set_grad_enabled(False)
    a.out.mkdir(parents=True, exist_ok=True)
    encoder, probe = load_l1_dense_checkpoint(a.l1_checkpoint, a.device)
    # Restore RNG state to avoid changing unrelated code randomness.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(a.random_seed)
        random_encoder = LocalStream('l1', 'rigid').to(a.device).eval()
    for param in random_encoder.parameters():
        param.requires_grad_(False)
    result = evaluate(a.data, encoder, probe, random_encoder, device=a.device,
                      clips_per_sweep=a.clips_per_sweep, batch=a.batch,
                      num_queries=a.queries, radius=a.radius,
                      raw_patch=a.raw_patch, max_dz_mm=a.proxy_max_dz_mm)
    result.update({'protocol': 'VAL only; frozen pretrained / random encoders; no probe training',
                   'checkpoint': str(a.l1_checkpoint), 'data': str(a.data),
                   'settings': {'queries': a.queries, 'feature_search_radius': a.radius,
                                'raw_patch': a.raw_patch, 'proxy_max_dz_mm': a.proxy_max_dz_mm,
                                'random_seed': a.random_seed},
                   'cautions': [
                       'Small |dz| is only a proxy for physical correspondence visibility.',
                       'Trained motion head uses supervision from L1 pretraining: not transfer evidence.',
                       'Feature matching is zero-shot cosine NN; failure does not rule out a learnable query decoder.',
                       'Feature matching quantized at stride 4; do not compare EPE without that caveat.',
                       'All methods share query samples, FOV and |dz| support within each gap.',
                       'Validation CT count is small; do not treat query points as independent subjects.',
                   ]})
    output = a.out / 'frozen_transfer_val.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'VAL: {result["n_ct"]} CT, {result["n_sweeps"]} sweeps, TEST untouched')
    for gap in GAPS:
        data = result['gaps'][str(gap)]
        small = data['micro'].get('proxy_overlap', {})
        names = ' '.join(f'{key}={small[key]["epe_px"]:.2f}px' for key in MODES if key in small)
        print(f'gap={gap} clips={data["n_clips"]} proxy_n={small.get("n_points",0)} {names}', flush=True)
    print(f'Report: {output}; encoder/head untouched; no training.', flush=True)


if __name__ == '__main__':
    main()
