"""Train/evaluate dense Local Stream heads on case-disjoint SonoGym sweeps.

--head dz (default) preserves the original |dz| experiment and checkpoints.
--head xyz predicts (dx,dy,|dz|) for combined 3D probe motion.
--eval-only --ncc-ablation isolates the NCC contribution with L1 frames fixed.
Probe poses and spacing are LABEL-ONLY; no test split access.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct2eco.local_data import SweepWindows
from ct2eco.local_ncc import spatial_ncc, dense_abs_z_from_poses, dense_motion_from_poses
from ct2eco.local_stream import LocalStream


MOTIONS = {'elevational_pure': 'elev_mag', 'rot_x': 'rot_x', 'rot_y': 'rot_y',
           'random': 'rigid'}


class DenseDzHead(nn.Module):
    """A small fully convolutional disposable probe: dense features -> mm."""
    def __init__(self, channels=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, 64, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(64, 1, 1), nn.Softplus())

    def forward(self, dense):
        b, n, c, h, w = dense.shape
        return self.net(dense.flatten(0, 1)).reshape(b, n, 1, h, w)



class DenseXYZHead(nn.Module):
    """(dx,dy,|dz|) dense mm head; last component constrained nonnegative."""
    def __init__(self, channels=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, 64, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(64, 3, 1))

    def forward(self, dense):
        b, n, c, h, w = dense.shape
        x = self.net(dense.flatten(0, 1)).reshape(b, n, 3, h, w)
        return torch.cat([x[:, :, :2], F.softplus(x[:, :, 2:3])], dim=2)


def loss_dense(pred, gt, mask, head_type):
    """Component-normalized Huber (xyz) or historical mm Huber (dz)."""
    if head_type == 'dz':
        field = F.smooth_l1_loss(pred, gt, beta=0.4, reduction='none')
    else:
        scale = pred.new_tensor([0.8, 0.8, 0.8])[None, None, :, None, None]
        field = F.smooth_l1_loss(pred / scale, gt / scale, beta=0.5, reduction='none')
    return masked_mean(field, mask.expand_as(field))

def supervision(video, poses, spacing, hw, valid_from_model=None, windows=(15, 31), head_type="dz"):
    if valid_from_model is None:
        _, mask = spatial_ncc(video, windows, out_hw=hw)
    else:
        mask = valid_from_model
    valid = mask.all(dim=2, keepdim=True)
    if head_type == 'xyz':
        target, overlap = dense_motion_from_poses(poses, video.shape[-2:], spacing, hw,
                                                   return_fov=True)
        return target, valid & overlap
    target = dense_abs_z_from_poses(poses, video.shape[-2:], spacing, hw)
    return target, valid


def masked_mean(values, mask):
    if not mask.any():
        raise ValueError('No valid speckle cells: inspect dataset and NCC min_variance')
    return values[mask].mean()


def make_loader(args, split, length, seed, shuffle=False):
    ds = SweepWindows(args.data, split, task=args.target_task, frames=args.frames,
                      length=length, seed=seed, expected_motion=args.motion, return_spacing=True)
    return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=args.workers,
                      drop_last=True, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def evaluate(encoder, head, batches, spatial_prior, max_mm, device, head_type='dz',
             ncc_ablation=False):
    encoder.eval()
    head.eval()
    channels = 3 if head_type == 'xyz' else 1
    keys = ['normal', 'zeros', 'other', 'prior', 'full_normal', 'full_prior']
    if ncc_ablation:
        if encoder.arch != 'l2b':
            raise ValueError('NCC ablation requires arch=l2b')
        keys.extend(['ncc_zero', 'ncc_other', 'ncc_spatial_roll'])
    sums = {k: torch.zeros(channels, dtype=torch.float64) for k in keys}
    counts = {k: 0 for k in keys}
    for video, _, poses, spacing in batches:
        x, p, sp = video.to(device), poses.to(device), spacing.to(device)
        encoded = encoder(x)
        dense = encoded['dense']
        target, valid = supervision(x, p, sp, dense.shape[-2:], encoded['ncc_valid'],
                                    head_type=head_type)
        reliable = valid & (target[:, :, -1:] <= max_mm)
        preds = {'normal': head(dense),
                 'zeros': head(encoder(torch.zeros_like(x))['dense']),
                 'other': head(encoder(x.roll(1, 0))['dense']),
                 'prior': spatial_prior.expand_as(target)}
        if ncc_ablation:
            if x.shape[0] < 2:
                raise ValueError('NCC other-video control requires batches with >=2 clips')
            dec = ((1 - encoded['ncc']) * encoded['ncc_valid'].float() * 10).detach()
            other_ncc, other_valid = spatial_ncc(x.roll(1, 0), out_hw=dense.shape[-2:])
            # Only the NCC input is altered; L1 frames and validity mask remain
            # those of the correct video. The spatial roll breaks alignment.
            alt = {'ncc_zero': torch.zeros_like(dec),
                   'ncc_other': ((1 - other_ncc) * other_valid.float() * 10).detach(),
                   'ncc_spatial_roll': dec.roll(shifts=(max(1, dec.shape[-2] // 2),
                                                          max(1, dec.shape[-1] // 2)),
                                                   dims=(-2, -1))}
            for name, override in alt.items():
                preds[name] = head(encoder(x, ncc_dec_override=override)['dense'])
        for name, prediction in preds.items():
            mask = reliable
            abs_err = (prediction - target).abs().double() * mask.double()
            sums[name] += abs_err.sum(dim=(0, 1, 3, 4)).cpu()
            counts[name] += int(mask.sum())
        # Full-image diagnostics share the normal/prior predictions, but use
        # the untruncated valid support instead of censoring >max_mm.
        for name in ('full_normal', 'full_prior'):
            prediction = preds[name[5:]]
            sums[name] += ((prediction - target).abs().double() * valid.double()).sum(dim=(0, 1, 3, 4)).cpu()
            counts[name] += int(valid.sum())
    encoder.train()
    head.train()
    if not counts['normal']:
        raise ValueError('Validation has no valid NCC pixels within observability range')
    ans = {key + '_mae_mm': float((total / max(counts[key], 1)).mean())
           for key, total in sums.items()}
    if head_type == 'xyz':
        ans['component_names'] = ['dx_mm', 'dy_mm', 'abs_dz_mm']
        for key, total in sums.items():
            ans[f'component_{key}_mae_mm'] = (total / max(counts[key], 1)).tolist()
    ans['ratio_vs_prior'] = ans['normal_mae_mm'] / max(ans['prior_mae_mm'], 1e-10)
    ans['evaluated_pixels'] = counts['normal']
    ans['full_valid_pixels'] = counts['full_normal']
    return ans


def train_spatial_prior(loader, device, head_type, max_mm):
    total = count = None
    for x, _, poses, spacing in loader:
        with torch.no_grad():
            _, ncc_valid = spatial_ncc(x, out_hw=((x.shape[-2] + 3) // 4,
                                                  (x.shape[-1] + 3) // 4))
            gt, valid = supervision(x, poses, spacing, ncc_valid.shape[-2:],
                                    ncc_valid, head_type=head_type)
            ok = valid & (gt[:, :, -1:] <= max_mm)
            summed = (gt * ok).sum(dim=(0, 1), keepdim=True)
            samples = ok.sum(dim=(0, 1), keepdim=True)
            total = summed if total is None else total + summed
            count = samples if count is None else count + samples
    # Scalar per-channel fallbacks use TRAIN statistics only.
    global_counts = count.sum(dim=(-2, -1), keepdim=True).clamp(min=1)
    fallback = total.sum(dim=(-2, -1), keepdim=True) / global_counts
    prior = torch.where(count > 0, total / count.clamp(min=1), fallback)
    return prior.to(device)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--motion', choices=list(MOTIONS), required=True)
    p.add_argument('--arch', choices=['l1', 'l2b'], required=True)
    p.add_argument('--head', choices=['dz', 'xyz'], default='dz',
                   help='dz: scalar |dz| (legacy); xyz: dense dx,dy,|dz| for combined motion')
    p.add_argument('--eval-only', action='store_true', help='Evaluate an existing best.pt; no training')
    p.add_argument('--init', type=Path, help='Checkpoint for --eval-only')
    p.add_argument('--ncc-ablation', action='store_true',
                   help='In --eval-only, perturb NCC maps only, keeping L1 features fixed')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--frames', type=int, default=5)
    p.add_argument('--min-frames', type=int, help='Train on random clip lengths in [min-frames, frames]')
    p.add_argument('--frame-norm', action='store_true', help='Per-frame GroupNorm in L1')
    p.add_argument('--val-batches', type=int, default=16)
    p.add_argument('--eval-every', type=int, default=150)
    p.add_argument('--max-dz-mm', type=float, default=3.0)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if a.steps < 1 or a.val_batches < 1 or a.max_dz_mm <= 0 or a.eval_every < 1:
        p.error('steps, val-batches, eval-every and max-dz-mm must be positive')
    if a.min_frames is not None and not 2 <= a.min_frames <= a.frames:
        p.error('--min-frames must be in [2, --frames]')
    if a.motion == 'random' and a.head != 'xyz':
        p.error('Mixed random motion requires --head xyz')
    if a.ncc_ablation and (not a.eval_only or a.arch != 'l2b'):
        p.error('--ncc-ablation requires --eval-only --arch l2b')
    if a.eval_only and a.init is None:
        p.error('--eval-only requires --init path/to/best.pt')
    a.target_task = MOTIONS[a.motion]
    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    val = list(make_loader(a, 'val', a.val_batches * a.batch, 900001))
    prior_loader = make_loader(a, 'train', 64, 808017)
    prior = train_spatial_prior(prior_loader, device, a.head, a.max_dz_mm)
    encoder = LocalStream(a.arch, a.target_task, frame_norm=a.frame_norm).to(device)
    # Identical probe initialization across L1 and L2b.
    torch.manual_seed(a.seed + 100003)
    head = (DenseXYZHead() if a.head == 'xyz' else DenseDzHead()).to(device)
    if a.eval_only:
        state = torch.load(a.init, map_location=device, weights_only=False)
        saved = state.get('config', {})
        if saved.get('head', 'dz') != a.head:
            raise ValueError('Head mismatch between checkpoint and --head')
        for name in ('motion', 'arch', 'frames'):
            if name in saved and saved[name] != getattr(a, name):
                raise ValueError(f'Checkpoint {name}={saved[name]} but requested {getattr(a, name)}')
        if 'max_dz_mm' in saved and saved['max_dz_mm'] != a.max_dz_mm:
            raise ValueError('To reproduce checkpoint validation, use its saved --max-dz-mm')
        encoder.load_state_dict(state['encoder'])
        head.load_state_dict(state['dense_probe'])
        metrics = evaluate(encoder, head, val, prior, a.max_dz_mm, device,
                           head_type=a.head, ncc_ablation=a.ncc_ablation)
        output = {'checkpoint': str(a.init), 'split': 'val', 'motion': a.motion,
                  'head': a.head, 'ncc_ablation': a.ncc_ablation, **metrics}
        (a.out / 'evaluation_val.json').write_text(json.dumps(output, indent=2) + '\n')
        print(json.dumps(output, indent=2), flush=True)
        print('Validation only; test split untouched.', flush=True)
        return
    train = make_loader(a, 'train', max(20000, a.steps * a.batch), a.seed, shuffle=True)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=a.lr, weight_decay=0.01)
    best = float('inf')
    history = a.out / 'history.jsonl'
    history.write_text('')
    batches = iter(train)
    for step in range(1, a.steps + 1):
        try:
            video, _, poses, spacing = next(batches)
        except StopIteration:
            batches = iter(train)
            video, _, poses, spacing = next(batches)
        x, ps, sp = video.to(device), poses.to(device), spacing.to(device)
        if a.min_frames is not None:
            n = int(torch.randint(a.min_frames, a.frames + 1, (1,)))
            s0 = int(torch.randint(0, a.frames - n + 1, (1,)))
            x, ps = x[:, s0:s0 + n], ps[:, s0:s0 + n]
        fraction = step / a.steps
        lr = a.lr * min(1, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * fraction)))
        for group in opt.param_groups:
            group['lr'] = lr
        encoded = encoder(x)
        pred = head(encoded['dense'])
        with torch.no_grad():
            gt, valid = supervision(x, ps, sp, pred.shape[-2:], encoded['ncc_valid'],
                                    head_type=a.head)
            mask = valid & (gt[:, :, -1:] <= a.max_dz_mm)
        loss = loss_dense(pred, gt, mask, a.head)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.)
        opt.step()
        if step == 1 or step % a.eval_every == 0 or step == a.steps:
            metrics = evaluate(encoder, head, val, prior, a.max_dz_mm, device, head_type=a.head)
            record = {'step': step, 'train_loss': float(loss.detach()), **metrics}
            with history.open('a') as f:
                f.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)
            checkpoint = {'encoder': encoder.state_dict(), 'dense_probe': head.state_dict(),
                          'step': step, 'config': {**vars(a), 'out': str(a.out), 'data': str(a.data)}}
            torch.save(checkpoint, a.out / 'last.pt')
            if metrics['normal_mae_mm'] < best:
                best = metrics['normal_mae_mm']
                torch.save(checkpoint, a.out / 'best.pt')
    print('Validation selected; test split not used.', flush=True)


if __name__ == '__main__':
    main()
