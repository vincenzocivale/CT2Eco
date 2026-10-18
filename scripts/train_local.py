"""Independent Local Stream diagnostic training for CT2Eco.

Usage (from CT2Eco root):
  python scripts/train_local.py --source phantom --task lateral --arch l0 --out runs/local_l0
  python scripts/train_local.py --source phantom --task lateral --arch l1 --out runs/local_l1
  python scripts/train_local.py --source phantom --task lateral --arch l2 --out runs/local_l2
  python scripts/train_local.py --source ct2eco --data data/pilot_v2 --task rigid --arch l1 --out runs/local_ct

No query decoder or global encoder is trained here. All pose labels are for
training diagnostics only, not model inputs. Saves best.pt and last.pt.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct2eco.local_data import SweepWindows, phantom_batch
from ct2eco.local_stream import LocalStream, regression_loss
from ct2eco.local_geometry import point_error_mm, point_loss


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', choices=['phantom', 'ct2eco'], default='phantom')
    p.add_argument('--data', type=Path, help='Required for --source ct2eco (generated CT2Eco dataset)')
    p.add_argument('--arch', choices=['l0', 'l1', 'l2', 'l2b'], default='l1')
    p.add_argument('--task', choices=list(LocalStream.TASK_DIMS), default='lateral')
    p.add_argument('--loss', choices=['huber', 'point'], default='huber',
                   help='point is SE(3) geometric point loss; only for rigid')
    p.add_argument('--select-metric', choices=['normal_mae', 'point_mae_mm'],
                   default='normal_mae', help='Validation checkpoint selection')
    p.add_argument('--eval-only', action='store_true', help='Evaluate --init without training')
    p.add_argument('--eval-split', choices=['val', 'test'], default='val',
                   help='test allowed ONLY with --eval-only; no selection on test')
    p.add_argument('--require-pure-motion', action='store_true',
                   help='Reject CT sweeps whose motion metadata is not the pure requested motion')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--frames', type=int, default=5)
    p.add_argument('--height', type=int, default=96)
    p.add_argument('--width', type=int, default=96)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--eval-every', type=int, default=150)
    p.add_argument('--val-batches', type=int, default=8)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--init', type=Path)
    p.add_argument('--overfit-opposites', action='store_true', help='Two fixed videos, SAME 3-D phantom, opposite lateral speeds; fit-only gate')
    p.add_argument('--freeze-encoder', action='store_true', help='Head-only linear-ish adaptation from --init')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def _expected_motion(args):
    if not args.require_pure_motion:
        return None
    options = {'lateral': 'lateral', 'elev_mag': 'elevational_pure',
               'elev_signed': 'elevational_pure', 'rot_x': 'rot_x', 'rot_y': 'rot_y',
               'inplane_rot': 'rot_z'}
    if args.task not in options:
        raise ValueError('--require-pure-motion is only available for isolated motion tasks')
    return options[args.task]


def make_validation(args, device):
    if args.source == 'phantom':
        gen = torch.Generator(device=device).manual_seed(900001 if args.eval_split == 'val' else 900002)
        samples = [phantom_batch(args.batch, args.frames, (args.height, args.width),
                                 args.task, device=device, generator=gen)
                   for _ in range(args.val_batches)]
        return [(a, b) for a, b, _ in samples]
    loader = DataLoader(SweepWindows(args.data, args.eval_split, args.task, args.frames,
                                     length=args.val_batches * args.batch, seed=900001,
                                     expected_motion=_expected_motion(args)),
                        batch_size=args.batch, num_workers=args.workers)
    return [(a.to(device), b.to(device)) for a, b, _ in loader]


def train_iterator(args, device):
    if args.source == 'phantom':
        gen = torch.Generator(device=device).manual_seed(args.seed)
        while True:
            images, targets, _ = phantom_batch(args.batch, args.frames, (args.height, args.width),
                                                args.task, device=device, generator=gen)
            yield images, targets
    else:
        ds = SweepWindows(args.data, 'train', args.task, args.frames,
                          length=max(20000, args.steps * args.batch), seed=args.seed,
                          expected_motion=_expected_motion(args))
        loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed), drop_last=True)
        while True:
            for x, y, _ in loader:
                yield x.to(device), y.to(device)


def prior_from_train(args, device):
    # Use training labels only; never estimate priors from validation labels.
    g = train_iterator(args, device)
    return torch.cat([next(g)[1].reshape(-1, LocalStream.TASK_DIMS[args.task])
                      for _ in range(16)]).mean(0)


def _rigid_reverse_expected(pred):
    """Analytic SE(3) inverse of a predicted rotvec + translation (diagnostic)."""
    w, t = pred[..., :3], pred[..., 3:]
    wx, wy, wz = w.unbind(-1)
    o = torch.zeros_like(wx)
    skew = torch.stack([o, -wz, wy, wz, o, -wx, -wy, wx, o], dim=-1)
    skew = skew.reshape(*w.shape[:-1], 3, 3)
    rotation = torch.matrix_exp(skew)
    inv_t = -(rotation.transpose(-1, -2) @ t[..., None])[..., 0]
    return torch.cat([-w, inv_t], dim=-1)


@torch.no_grad()
def evaluate(model, validation, task, prior):
    """Image-ablated metrics with per-component breakdowns and physical 3D geometry.

    Important: corrupted-image errors use the ORIGINAL motion target, by design.
    time_reverse_mae measures prediction-inversion consistency, not GT accuracy.
    """
    model.eval()
    conditions = ('normal', 'zeros', 'other_video', 'time_shuffle')
    records = {name: [] for name in conditions}
    records['time_reverse'] = []
    scales = (torch.tensor([0.04, 0.04, 0.04, 0.8, 0.8, 0.8], device=prior.device)
              if task == 'rigid' else torch.ones_like(prior))
    flat_preds = {name: [] for name in conditions}
    tars = []
    for x, y in validation:
        pred = model(x)['prediction']
        zero_pred = model(torch.zeros_like(x))['prediction']
        other_pred = model(x.roll(1, 0))['prediction'] if x.shape[0] > 1 else zero_pred
        reverse_pred = model(x.flip(1))['prediction'].flip(1)
        perm = torch.cat([torch.arange(0, x.shape[1], 2, device=x.device),
                          torch.arange(1, x.shape[1], 2, device=x.device)])
        shuffled_pred = model(x[:, perm])['prediction']
        for name, estimate in zip(conditions, (pred, zero_pred, other_pred, shuffled_pred)):
            records[name].append(((estimate - y).abs() / scales).mean(dim=(1, 2)).cpu())
            flat_preds[name].append(estimate.detach().cpu().reshape(-1, y.shape[-1]))
        expected = (pred if task == 'elev_mag' else
                    _rigid_reverse_expected(pred) if task == 'rigid' else -pred)
        records['time_reverse'].append(((reverse_pred - expected).abs() / scales).mean(dim=(1, 2)).cpu())
        tars.append(y.detach().cpu().reshape(-1, y.shape[-1]))
    gt = torch.cat(tars)
    pd = {name: torch.cat(parts) for name, parts in flat_preds.items()}
    prior_cpu = prior.detach().cpu()
    scale_cpu = scales.detach().cpu()
    blind_mae = ((gt - prior_cpu).abs() / scale_cpu).mean().item()
    ans = {f'{key}_mae': torch.cat(values).mean().item() for key, values in records.items()}
    ans['blind_mae'] = blind_mae
    ans['ratio_vs_blind'] = ans['normal_mae'] / max(blind_mae, 1e-12)
    ans['per_component_blind_mae'] = (gt - prior_cpu).abs().mean(0).tolist()
    # Preserve existing names while adding ablation-specific component errors.
    for name in conditions:
        component_mae = (pd[name] - gt).abs().mean(0).tolist()
        ans[f'per_component_{name}_mae'] = component_mae
        if name == 'normal':
            ans['per_component_mae'] = component_mae
    if task in ('lateral', 'inplane_rot', 'rot_x', 'rot_y', 'elev_signed'):
        # Exclude near-zero GT from directional accuracy, as its sign is unstable.
        nonzero = gt.abs() > 1e-5
        ans['sign_accuracy'] = ((pd['normal'] * gt > 0) & nonzero).sum().item() / max(nonzero.sum().item(), 1)
    if task == 'rigid':
        for name in conditions:
            ans[f'point_{name}_mae_mm'] = point_error_mm(pd[name], gt).mean().item()
        ans['point_mae_mm'] = ans['point_normal_mae_mm']  # backward compatible
        ans['point_blind_mae_mm'] = point_error_mm(prior_cpu.expand_as(gt), gt).mean().item()
        ans['point_identity_mae_mm'] = point_error_mm(torch.zeros_like(gt), gt).mean().item()
        ans['rotation_vector_mae_rad'] = (pd['normal'][:, :3] - gt[:, :3]).norm(dim=1).mean().item()
        ans['translation_mae_mm'] = (pd['normal'][:, 3:] - gt[:, 3:]).norm(dim=1).mean().item()
    model.train()
    return ans


def main():
    a = arguments()
    if a.source == 'ct2eco' and a.data is None:
        raise SystemExit('--source ct2eco requires --data')
    if a.overfit_opposites and (a.source != 'phantom' or a.task != 'lateral' or a.batch != 2):
        raise SystemExit('--overfit-opposites requires --source phantom --task lateral --batch 2')
    if a.freeze_encoder and a.init is None:
        raise SystemExit('--freeze-encoder requires --init')
    if a.eval_only and a.init is None:
        raise SystemExit('--eval-only requires --init checkpoint')
    if a.eval_split == 'test' and not a.eval_only:
        raise SystemExit('Test split is locked for checkpoint selection: use --eval-only --init')
    if a.loss == 'point' and a.task != 'rigid':
        raise SystemExit('--loss point is only defined for --task rigid')
    if a.select_metric == 'point_mae_mm' and a.task != 'rigid':
        raise SystemExit('--select-metric point_mae_mm requires --task rigid')
    if a.require_pure_motion and a.source != 'ct2eco':
        raise SystemExit('--require-pure-motion requires --source ct2eco')
    if a.steps < 1 or a.eval_every < 1 or a.val_batches < 1:
        raise SystemExit('--steps, --eval-every, --val-batches must be positive')
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    model = LocalStream(a.arch, a.task).to(device)
    if a.init:
        obj = torch.load(a.init, map_location='cpu', weights_only=False)
        model.load_state_dict(obj['model'])
    if a.freeze_encoder:
        for par in model.encoder.parameters():
            par.requires_grad_(False)
        if model.correlation is not None:
            for par in model.correlation.parameters():
                par.requires_grad_(False)
    if not a.eval_only:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=a.lr, weight_decay=0.01)
    if a.overfit_opposites:
        gen = torch.Generator(device=device).manual_seed(a.seed)
        px = torch.tensor([-1.5, 1.5], device=device)
        xx, yy, _ = phantom_batch(2, a.frames, (a.height, a.width), 'lateral',
                                  device=device, generator=gen,
                                  signed_speed_px=px, shared_volume=True)
        def repeated():
            while True:
                yield xx, yy
        train_batches = repeated()
        validation = [(xx, yy)]  # deliberate FIT evaluation, not held-out
        prior = yy.mean(dim=(0, 1))
        print('OVERFIT-OPPOSITES: two fixed opposite-motion videos; validation == train.', flush=True)
    else:
        train_batches = train_iterator(a, device)
        validation = make_validation(a, device)
        prior = prior_from_train(a, device)
    if a.eval_only:
        metrics = evaluate(model, validation, a.task, prior)
        record = {'checkpoint': str(a.init), 'source': a.source, 'task': a.task,
                  'eval_split': a.eval_split, **metrics}
        (a.out / f'evaluation_{a.eval_split}.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record, indent=2), flush=True)
        return
    report_file = a.out / 'history.jsonl'
    report_file.write_text('')
    best = float('inf')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'arch={a.arch} task={a.task} params={n_params:,} source={a.source}', flush=True)
    for step in range(1, a.steps + 1):
        x, y = next(train_batches)
        frac = step / max(1, a.steps)
        warm = min(1.0, step / max(1, a.warmup))
        lr = a.lr * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for pg in opt.param_groups:
            pg['lr'] = lr
        model.train()
        pred = model(x)['prediction']
        loss = point_loss(pred, y) if a.loss == 'point' else regression_loss(pred, y, a.task)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.eval_every == 0 or step in (1, a.steps):
            vals = evaluate(model, validation, a.task, prior)
            record = {'step': step, 'train_loss': float(loss.detach()), 'grad_norm': float(grad_norm),
                      'lr': lr, **vals}
            with report_file.open('a') as f:
                f.write(json.dumps(record) + '\n')
            print(json.dumps({k: round(v, 5) if isinstance(v, float) else v
                              for k, v in record.items()}), flush=True)
            ckpt = {'model': model.state_dict(), 'step': step, 'config': vars(a) | {'out': str(a.out), 'data': str(a.data)}}
            torch.save(ckpt, a.out / 'last.pt')
            if vals[a.select_metric] < best:
                best = vals[a.select_metric]
                torch.save(ckpt, a.out / 'best.pt')
    print('Checkpoint selection used validation; run confirmatory analysis on held-out CT/test data.', flush=True)


if __name__ == '__main__':
    main()
