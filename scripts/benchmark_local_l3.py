"""Paired inference-only R0/R2 latency benchmark on one real validation clip batch.

Measures forward(video, spacing) only, excluding checkpoint loading, disk I/O,
GT-label creation, and metrics; GT poses are never passed to either model.
Note: R0 still computes fixed NCC for the common supervision validity mask;
latencies describe the *current implementation*, not an optimized R0.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct2eco.local_l1_alignment import load_l1_dense_checkpoint
from ct2eco.local_l3_residual import LocalL3Residual, DzResidual
from scripts.train_local_dense import make_loader
from scripts.train_local_l3 import sha256, verify_data_provenance


def load_residual(path, expected_variant, teacher_hash, root):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    for key, value in {'variant': expected_variant, 'frames': 5,
                       'motion': 'random', 'head': 'xyz',
                       'dataset': str(root.resolve()), 'teacher_sha256': teacher_hash}.items():
        if cfg.get(key) != value:
            raise ValueError(f'Checkpoint {path}: config mismatch for {key}: {cfg.get(key)} != {value}')
    return ckpt


def timed_forward(model, x, spacing, warmup, iters):
    def sync():
        if x.device.type == 'cuda':
            torch.cuda.synchronize(x.device)
    with torch.inference_mode():
        for _ in range(warmup):
            output = model(x, spacing)
        del output
        sync()
        if x.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(x.device)
        times = []
        for _ in range(iters):
            sync()
            start = time.perf_counter()
            output = model(x, spacing)
            sync()
            times.append((time.perf_counter() - start) * 1000)
            del output
        peak_mb = (torch.cuda.max_memory_allocated(x.device) / (1024 * 1024)
                   if x.device.type == 'cuda' else None)
    return {'median_batch_ms': statistics.median(times),
            'mean_batch_ms': statistics.mean(times),
            'median_clip_ms': statistics.median(times) / len(x),
            'gpu_peak_allocated_mib': peak_mb}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--l1-checkpoint', type=Path, required=True)
    p.add_argument('--r0', type=Path, required=True)
    p.add_argument('--r2', type=Path, required=True)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', type=Path, default=Path('runs/local/l3_inference_cost.json'))
    args = p.parse_args()
    if min(args.batch, args.warmup, args.iters) < 1:
        p.error('batch, warmup, iters must be positive')
    verify_data_provenance(args.data, args.l1_checkpoint)
    teacher_hash = sha256(args.l1_checkpoint)
    checkpoints = {arm: load_residual(path, arm, teacher_hash, args.data)
                   for arm, path in [('r0', args.r0), ('r2', args.r2)]}
    c0, c2 = checkpoints['r0']['config'], checkpoints['r2']['config']
    if {k: v for k, v in c0.items() if k != 'variant'} != {k: v for k, v in c2.items() if k != 'variant'}:
        raise ValueError('Benchmark requires matched R0/R2 configurations (except variant)')
    if c0['windows'] != c2['windows']:
        raise ValueError('NCC windows must match')
    device = torch.device(args.device)
    encoder, probe = load_l1_dense_checkpoint(args.l1_checkpoint, device)
    class LoaderArgs:
        pass
    opts = LoaderArgs()
    opts.data, opts.target_task, opts.motion = args.data, 'rigid', 'random'
    opts.batch, opts.frames, opts.workers = args.batch, 5, 0
    loader = make_loader(opts, 'val', args.batch, 900001)
    video, _, _, spacing = next(iter(loader))
    x, sp = video.to(device), spacing.to(device)
    metrics = {}
    for arm in ('r0', 'r2'):
        model = LocalL3Residual(encoder, probe, arm, windows=c0['windows']).to(device).eval()
        model.refiner = DzResidual().to(device)
        model.refiner.load_state_dict(checkpoints[arm]['residual'], strict=True)
        metrics[arm] = timed_forward(model, x, sp, args.warmup, args.iters)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    cost = {'source_split': 'val', 'n_clips': args.batch,
            'frame_shape': list(x.shape), 'device': str(device),
            'warmup': args.warmup, 'iterations': args.iters,
            'r0': metrics['r0'], 'r2': metrics['r2'],
            'r2_over_r0_median_latency': metrics['r2']['median_batch_ms'] / metrics['r0']['median_batch_ms'],
            'note': 'R0 computes fixed NCC for validity even without using it; timings are implementation-specific.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cost, indent=2) + '\n')
    print(json.dumps(cost, indent=2))
    print('Forward-only benchmark; no TEST access.')


if __name__ == '__main__':
    main()
