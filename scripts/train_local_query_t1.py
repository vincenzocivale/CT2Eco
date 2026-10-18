"""Gate T1: learned point-query decoder on frozen L1 video representations.

Q0: query coordinates+gap only (no images); Q1: frozen RANDOM L1;
Q2: frozen pretrained L1. Identical readout capacity, initialization, query
samples and validation windows. Q3 (existing supervised L1 motion head) is a
validation-only XY reference, not an independent trained probe; no signed dz
is available for reliable multi-step composition.

GT poses enter only project_queries() to create targets. Neither frozen
encoder nor query decoder can receive poses. TEST is never evaluated.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct2eco.local_data import SweepWindows
from ct2eco.local_l1_alignment import load_l1_dense_checkpoint
from ct2eco.local_query_t1 import (GAPS, PointQueryReadout, aggregate_query_errors,
                                    finalize_query_errors, loss_point_queries)
from ct2eco.local_stream import FrameGroupNorm, LocalStream
from ct2eco.local_transfer import (compose_l1_displacements, project_queries,
                                   query_positions)
from scripts.eval_local_ncc_alignment import _sweep_paths
from scripts.train_local_l3 import verify_data_provenance


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for blob in iter(lambda: handle.read(1048576), b''):
            h.update(blob)
    return h.hexdigest()


def make_frozen_encoder(arm, teacher, device, random_seed=314159):
    if arm == 'q0':
        return None
    if arm == 'q2':
        return teacher
    if arm != 'q1':
        raise ValueError(arm)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(random_seed)
        random_encoder = LocalStream('l1', 'rigid', frame_norm=isinstance(
            teacher.encoder.net[1], FrameGroupNorm)).to(device).eval()
    for p in random_encoder.parameters():
        p.requires_grad_(False)
    return random_encoder


def decode(model, frozen_encoder, video, queries, gap):
    # Frozen features have no gradient path and cannot adapt to T1 labels.
    with torch.no_grad():
        features = None if frozen_encoder is None else frozen_encoder(video)
    return model(queries, gap, video.shape[-2:], len(video), features)


def _zero_z_reference(xy):
    return torch.cat([xy, xy.new_zeros(*xy.shape[:-1], 1)], dim=-1)


@torch.no_grad()
def evaluate(data, model, frozen_encoder, teacher, motion_head, device,
             clips_per_sweep=8, batch=4, queries_per_clip=64, overlap_mm=.5):
    model.eval()
    paths = _sweep_paths(data, 'val')  # asserts case-disjoint splits + random-motion provenance
    by_gap, by_ct = {}, {}
    clips = {}
    for gap in GAPS:
        by_gap[str(gap)] = {'query': {}, 'identity': {}, 'q3_supervised_xy': {}}
        by_ct[str(gap)] = {}
        clips[str(gap)] = 0
    for folder, meta in paths:
        frames = np.load(folder / 'frames.npy', mmap_mode='r')
        poses = np.load(folder / 'poses.npy', mmap_mode='r')
        if len(frames) != len(poses) or len(frames) < 5:
            raise ValueError(f'Invalid clip/pose length for {folder}')
        starts = np.unique(np.linspace(0, len(frames)-5, min(clips_per_sweep, len(frames)-4)).astype(int))
        case = str(meta['case_id'])
        for index in range(0, len(starts), batch):
            locs = starts[index:index+batch]
            video = torch.from_numpy(np.stack([np.asarray(frames[t:t+5]) for t in locs]).copy()).to(device).float()[:, :, None] / 255
            pp = torch.from_numpy(np.stack([np.asarray(poses[t:t+5]) for t in locs]).copy()).to(device).float()
            spacing = torch.as_tensor(meta['spacing_mm'], device=device, dtype=torch.float32).expand(len(locs), 2)
            if frozen_encoder is None:
                h, w = (video.shape[-2]+3)//4, (video.shape[-1]+3)//4
            else:
                current_features = frozen_encoder(video)
                h, w = current_features['dense'].shape[-2:]
            queries = query_positions((h,w), queries_per_clip, device=device)
            if frozen_encoder is None:
                current_features = None
            teacher_features = teacher(video) if frozen_encoder is not teacher else current_features
            teacher_motion = motion_head(teacher_features['dense'])
            source_uv = queries.float()[None].expand(len(locs), -1, -1) * 4
            for gap in GAPS:
                target_uv, abs_z, in_fov = project_queries(pp, spacing, queries,
                                                           video.shape[-2:], 0, gap)
                prediction = model(queries, gap, video.shape[-2:], len(video), current_features)
                xy_q3 = compose_l1_displacements(teacher_motion, queries, spacing, gap)
                predicted = {'query': prediction,
                             'identity': _zero_z_reference(source_uv),
                             'q3_supervised_xy': _zero_z_reference(xy_q3)}
                key = str(gap)
                case_store = by_ct[key].setdefault(case, {name: {} for name in predicted})
                for name, estimate in predicted.items():
                    aggregate_query_errors(by_gap[key][name], estimate, target_uv, abs_z,
                                           in_fov, overlap_mm)
                    aggregate_query_errors(case_store[name], estimate, target_uv, abs_z,
                                           in_fov, overlap_mm)
                clips[key] += len(locs)
    report = {'n_ct': len(set(str(meta['case_id']) for _, meta in paths)),
              'n_sweeps': len(paths), 'gaps': {}}
    for key in by_gap:
        methods = {name: finalize_query_errors(values) for name, values in by_gap[key].items()}
        cases = {case: {name: finalize_query_errors(val) for name, val in methods_.items()}
                 for case, methods_ in by_ct[key].items()}
        macro = {}
        for name in ('query','identity','q3_supervised_xy'):
            measurements = [methods_[name]['fov_all']['epe_px'] for methods_ in cases.values()
                            if 'fov_all' in methods_[name]]
            macro[name] = float(np.mean(measurements)) if measurements else None
        # Q3 is a supervised *XY* baseline; elevation is not estimated for gaps>1.
        for container in [methods, *(cases.values())]:
            for band in container['q3_supervised_xy'].values():
                band.pop('abs_dz_mae_mm', None)
        report['gaps'][key] = {'n_clips': clips[key], 'micro': methods, 'by_ct': cases,
                               'macro_ct_fov_epe_px': macro}
    return report


def settings(args):
    return {'variant': args.variant, 'data': str(args.data.resolve()),
            'l1_hash': sha256(args.l1_checkpoint),
            'seed': args.seed, 'train_window_seed': args.train_window_seed,
            'random_encoder_seed': args.random_encoder_seed,
            'queries': args.queries, 'val_clips_per_sweep': args.val_clips_per_sweep,
            'overlap_mm': args.overlap_mm, 'training_gaps': list(GAPS),
            'query_frame': 0, 'frames': 5}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--l1-checkpoint', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--variant', required=True, choices=['q0', 'q1', 'q2'])
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--queries', type=int, default=64)
    p.add_argument('--val-clips-per-sweep', type=int, default=8)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--train-window-seed', type=int, default=17001)
    p.add_argument('--random-encoder-seed', type=int, default=314159)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--overlap-mm', type=float, default=.5)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--init', type=Path)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if any(x < 1 for x in (a.steps,a.batch,a.queries,a.val_clips_per_sweep,a.eval_every)) or a.overlap_mm <= 0 or a.lr <= 0:
        p.error('Positive steps/batch/queries/eval-every/overlap/lr required')
    if a.eval_only != (a.init is not None):
        p.error('--eval-only requires --init; --init is for --eval-only')
    verify_data_provenance(a.data, a.l1_checkpoint)
    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    teacher, motion_head = load_l1_dense_checkpoint(a.l1_checkpoint, device)
    frozen = make_frozen_encoder(a.variant, teacher, device, a.random_encoder_seed)
    torch.manual_seed(a.seed + 100003)  # exactly the same decoder init in all three arms
    decoder = PointQueryReadout().to(device)
    config = settings(a)
    if a.eval_only:
        ckpt = torch.load(a.init, map_location=device, weights_only=False)
        if ckpt.get('config') != config:
            raise ValueError('T1 checkpoint settings/provenance do not match this evaluation')
        decoder.load_state_dict(ckpt['decoder'], strict=True)
        result = evaluate(a.data, decoder, frozen, teacher, motion_head, device,
                          a.val_clips_per_sweep, a.batch, a.queries, a.overlap_mm)
        result.update({'config': config, 'checkpoint_step': ckpt['step'], 'split': 'val'})
        (a.out / 'evaluation_val.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'variant': a.variant, 'step': ckpt['step'],
                          'gap_macro': {k:v['macro_ct_fov_epe_px'] for k,v in result['gaps'].items()}}, indent=2))
        return
    train = SweepWindows(a.data, 'train', task='rigid', frames=5,
                         length=max(a.steps * a.batch, 8192), seed=a.train_window_seed,
                         expected_motion='random', return_spacing=True)
    loader = DataLoader(train, batch_size=a.batch, shuffle=True, drop_last=True,
                        num_workers=a.workers,
                        generator=torch.Generator().manual_seed(a.seed+111))
    generator = torch.Generator().manual_seed(a.seed+222)
    opt = torch.optim.AdamW(decoder.parameters(), lr=a.lr, weight_decay=.01)
    best = float('inf')
    (a.out / 'history.jsonl').write_text('')
    batches = iter(loader)
    for step in range(1, a.steps + 1):
        try:
            video, _, poses, spacing = next(batches)
        except StopIteration:
            batches = iter(loader)
            video, _, poses, spacing = next(batches)
        video, poses, spacing = video.to(device), poses.to(device), spacing.to(device)
        gap = GAPS[int(torch.randint(len(GAPS), (1,), generator=generator))]
        fh, fw = (video.shape[-2]+3)//4, (video.shape[-1]+3)//4
        q = query_positions((fh,fw), a.queries, device=device)
        # GT is strictly labels. It is never passed into decode() or the network.
        target_uv, abs_z, in_fov = project_queries(poses, spacing, q, video.shape[-2:], 0, gap)
        decoder.train()
        prediction = decode(decoder, frozen, video, q, gap)
        loss = loss_point_queries(prediction, target_uv, abs_z, in_fov)
        lr = a.lr * min(1., step/100) * (.1 + .9 * .5 * (1+math.cos(math.pi*step/a.steps)))
        for group in opt.param_groups:
            group['lr'] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.)
        opt.step()
        if step == 1 or step % a.eval_every == 0 or step == a.steps:
            result = evaluate(a.data, decoder, frozen, teacher, motion_head, device,
                              a.val_clips_per_sweep, a.batch, a.queries, a.overlap_mm)
            # Equal CT weighting, averaged across the three gaps.
            macro = [result['gaps'][str(g)]['macro_ct_fov_epe_px']['query'] for g in GAPS]
            primary = float(np.mean(macro))
            record = {'step':step, 'variant':a.variant, 'train_loss': float(loss.detach()),
                      'macro_ct_epe_px':primary,
                      'gaps': {str(g): {'query': result['gaps'][str(g)]['micro']['query'],
                                        'macro_epe': result['gaps'][str(g)]['macro_ct_fov_epe_px']['query']}
                               for g in GAPS}}
            with (a.out / 'history.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            print(f'{a.variant} step={step} train_loss={float(loss.detach()):.4f} '
                  f'macro_epe={primary:.3f}px gaps=' + ','.join(f'{x:.3f}' for x in macro), flush=True)
            saved = {'decoder': decoder.state_dict(), 'step':step, 'config':config}
            torch.save(saved, a.out / 'last.pt')
            if primary < best:
                best = primary
                torch.save(saved, a.out / 'best.pt')
                result.update({'config':config, 'checkpoint_step':step, 'split':'val',
                               'selection':'mean across gaps of CT-macro in-FOV XY EPE'})
                (a.out / 'best_val.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'T1 {a.variant}: best validation macro EPE={best:.4f}px; test untouched.')


if __name__ == '__main__':
    main()
