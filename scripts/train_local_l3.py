"""Train a frozen-L1 dz-only residual on mixed SonoGym motion: R0/R1/R2.

All variants have equal head capacity, shared deterministic case-disjoint val
windows, identical pose-derived supervision masks and identical teacher motion
in the plane. No TEST data access. Checkpoints store ONLY the residual head.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct2eco.local_l1_alignment import load_l1_dense_checkpoint
from ct2eco.local_l3_residual import LocalL3Residual
from scripts.train_local_dense import make_loader, supervision


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sampled_case_ids(dataset):
    """Mirror deterministic index->sweep selection for per-CT validation stats.

    This is metric bookkeeping only. The sample CT ID is NEVER passed to the
    model or residual head. Assumes DataLoader shuffling is disabled for val.
    """
    case_map = [json.loads((entry / 'sweep.json').read_text())['case_id']
                for entry in dataset.entries]
    return [case_map[int(np.random.default_rng(dataset.seed + i)
                         .integers(len(dataset.entries)))] for i in range(len(dataset))]


def combined_mask(x, poses, spacing, output_hw, fixed_valid, max_dz_mm):
    target, valid = supervision(x, poses, spacing, output_hw, fixed_valid, head_type='xyz')
    return target, valid & (target[:, :, 2:3] <= max_dz_mm), valid


def dz_loss(pred, target, mask):
    if not bool(mask.any()):
        raise ValueError('Empty training support for dz; inspect masks and dataset')
    return (F.smooth_l1_loss(pred / .8, target / .8, beta=.5,
                             reduction='none') * mask).sum() / mask.sum()


@torch.no_grad()
def evaluate(model, loader, max_dz_mm, device, case_ids=None):
    model.eval()
    sums = {'base': torch.zeros(3, dtype=torch.float64),
            'refined': torch.zeros(3, dtype=torch.float64),
            'full_base': torch.zeros(3, dtype=torch.float64),
            'full_refined': torch.zeros(3, dtype=torch.float64)}
    counts = {k: 0 for k in sums}
    by_ct = {}
    visited = 0
    for video, _, poses, spacing in loader:
        x, ps, sp = video.to(device), poses.to(device), spacing.to(device)
        out = model(x, sp)
        gt, primary, full = combined_mask(x, ps, sp, out['prediction'].shape[-2:],
                                           out['fixed_valid'], max_dz_mm)
        samples = len(x)
        for key, prediction, mask in (
            ('base', out['baseline'], primary),
            ('refined', out['prediction'], primary),
            ('full_base', out['baseline'], full),
            ('full_refined', out['prediction'], full),
        ):
            err = (prediction - gt).abs().double() * mask.double()
            sums[key] += err.sum(dim=(0, 1, 3, 4)).cpu()
            counts[key] += int(mask.sum())
        if case_ids is not None:
            for i in range(samples):
                cid = case_ids[visited + i]
                stats = by_ct.setdefault(cid, {'base_sum': 0., 'refined_sum': 0., 'pixels': 0})
                m = primary[i, :, 0]
                stats['pixels'] += int(m.sum())
                stats['base_sum'] += float(((out['baseline'][i, :, 2] - gt[i, :, 2]).abs() * m).sum())
                stats['refined_sum'] += float(((out['prediction'][i, :, 2] - gt[i, :, 2]).abs() * m).sum())
        visited += samples
    if counts['base'] == 0:
        raise ValueError('No valid validation pixels; check max_dz_mm and window size')
    report = {'variant': model.variant,
              'primary_pixels': counts['base'], 'full_valid_pixels': counts['full_base'],
              'component_names': ['dx_mm', 'dy_mm', 'abs_dz_mm']}
    for key, value in sums.items():
        report[f'{key}_component_mae_mm'] = (value / max(1, counts[key])).tolist()
        report[f'{key}_dz_mae_mm'] = float(value[2] / max(1, counts[key]))
        report[f'{key}_mean_mae_mm'] = float(value.mean() / max(1, counts[key]))
    report['delta_dz_vs_l1_mm'] = report['refined_dz_mae_mm'] - report['base_dz_mae_mm']
    report['by_ct'] = {case: {
        'pixels': s['pixels'], 'base_dz_mae_mm': s['base_sum'] / s['pixels'],
        'refined_dz_mae_mm': s['refined_sum'] / s['pixels'],
        'delta_dz_vs_l1_mm': (s['refined_sum'] - s['base_sum']) / s['pixels']
    } for case, s in sorted(by_ct.items()) if s['pixels']}
    report['n_ct'] = len(report['by_ct']) if case_ids is not None else None
    return report


def verify_data_provenance(data_root, teacher_checkpoint):
    """Reject overlapping case splits and a known checkpoint/dataset mismatch."""
    splits = json.loads((data_root / 'metadata' / 'splits.json').read_text())
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if set(splits[a]) & set(splits[b]):
            raise ValueError(f'Case split leakage between {a} and {b}')
    # Old checkpoints lacking dataset provenance remain supported, but cannot
    # establish identity automatically. This run still requires case-disjoint data.
    teacher = torch.load(teacher_checkpoint, map_location='cpu', weights_only=False)
    prior_data = teacher.get('config', {}).get('data')
    if prior_data is not None and Path(prior_data).resolve() != data_root.resolve():
        raise ValueError(f'L1 checkpoint was trained on {prior_data}, but requested {data_root}')


def checkpoint_config(args, teacher_hash):
    return {'variant': args.variant, 'frames': args.frames, 'motion': args.motion,
            'head': 'xyz', 'max_dz_mm': args.max_dz_mm, 'windows': list(args.windows),
            'dataset': str(args.data.resolve()), 'teacher_sha256': teacher_hash,
            'seed': args.seed}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--l1-checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--variant', choices=['r0', 'r1', 'r2'], required=True)
    p.add_argument('--motion', choices=['random'], default='random')
    p.add_argument('--frames', type=int, default=5)
    p.add_argument('--windows', type=int, nargs=2, default=[15, 31])
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--val-batches', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=150)
    p.add_argument('--max-dz-mm', type=float, default=3.0)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--init', type=Path, help='Saved residual checkpoint for --eval-only')
    args = p.parse_args()
    if (args.frames != 5 or args.batch < 1 or args.val_batches < 1 or args.steps < 1
            or args.eval_every < 1 or args.max_dz_mm <= 0 or args.lr <= 0):
        p.error('Invalid training settings (L1 requires exactly 5 frames)')
    if args.eval_only != (args.init is not None):
        p.error('Use --eval-only together with --init path/to/checkpoint')
    args.target_task = 'rigid'
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    verify_data_provenance(args.data, args.l1_checkpoint)
    args.out.mkdir(parents=True, exist_ok=True)
    encoder, probe = load_l1_dense_checkpoint(args.l1_checkpoint, device=device)
    teacher_hash = sha256(args.l1_checkpoint)
    model = LocalL3Residual(encoder, probe, args.variant, windows=args.windows).to(device)
    torch.manual_seed(args.seed + 77771)  # same residual initialization across R0/R1/R2
    # Reinitialize the residual while leaving the loaded L1 teacher unchanged.
    from ct2eco.local_l3_residual import DzResidual
    model.refiner = DzResidual().to(device)
    config = checkpoint_config(args, teacher_hash)
    val_loader = make_loader(args, 'val', args.val_batches * args.batch, 900001)
    case_ids = sampled_case_ids(val_loader.dataset)
    if args.eval_only:
        saved = torch.load(args.init, map_location=device, weights_only=False)
        if saved['config'] != config:
            raise ValueError(f'Residual checkpoint/config mismatch: {saved["config"]} != {config}')
        model.refiner.load_state_dict(saved['residual'], strict=True)
        result = evaluate(model, val_loader, args.max_dz_mm, device, case_ids)
        result.update({'split': 'val', 'step': saved['step'], 'teacher_sha256': teacher_hash,
                       'residual_checkpoint': str(args.init)})
        (args.out / 'evaluation_val.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)
        return
    train_loader = make_loader(args, 'train', max(20000, args.steps * args.batch), args.seed,
                               shuffle=True)
    opt = torch.optim.AdamW(model.refiner.parameters(), lr=args.lr, weight_decay=.01)
    best = float('inf')
    history = args.out / 'history.jsonl'
    history.write_text('')
    batches = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            video, _, poses, spacing = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            video, _, poses, spacing = next(batches)
        x, ps, sp = video.to(device), poses.to(device), spacing.to(device)
        fraction = step / args.steps
        lr = args.lr * min(1., step / 100) * (.1 + .9 * .5 * (1 + math.cos(math.pi * fraction)))
        for group in opt.param_groups:
            group['lr'] = lr
        model.train()
        output = model(x, sp)
        with torch.no_grad():
            gt, mask, _ = combined_mask(x, ps, sp, output['prediction'].shape[-2:],
                                         output['fixed_valid'], args.max_dz_mm)
        loss = dz_loss(output['prediction'][:, :, 2:3], gt[:, :, 2:3], mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.refiner.parameters(), 1.)
        opt.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            result = evaluate(model, val_loader, args.max_dz_mm, device, case_ids)
            record = {'step': step, 'train_loss': float(loss.detach()), 'lr': lr, **result}
            with history.open('a') as f:
                f.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
            saved = {'residual': model.refiner.state_dict(), 'step': step, 'config': config}
            torch.save(saved, args.out / 'last.pt')
            if result['refined_dz_mae_mm'] < best:
                best = result['refined_dz_mae_mm']
                torch.save(saved, args.out / 'best.pt')
    print('Validation checkpoint selected by |dz| MAE; TEST untouched.', flush=True)


if __name__ == '__main__':
    main()
