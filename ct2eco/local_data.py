"""Diagnostic local-motion data: online 3-D scatterer phantoms and CT2Eco sweeps.

Targets are always computed from *known* probe-to-world poses; poses never enter
model.forward. Phantoms randomize signed motion independently from image texture.
"""

from pathlib import Path
import json

import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


def _rotations(angles):
    """B,T,3 XYZ Euler angles in radians -> B,T,3,3 (Rz @ Ry @ Rx)."""
    vx, vy, vz = angles.unbind(-1)
    zeros = torch.zeros_like(vx)
    ones = torch.ones_like(vx)
    cx, sx = vx.cos(), vx.sin()
    cy, sy = vy.cos(), vy.sin()
    cz, sz = vz.cos(), vz.sin()
    rx = torch.stack([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx], -1).reshape(*vx.shape, 3, 3)
    ry = torch.stack([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy], -1).reshape(*vx.shape, 3, 3)
    rz = torch.stack([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones], -1).reshape(*vx.shape, 3, 3)
    return rz @ ry @ rx


def _labels_from_poses(poses, task):
    """poses [B,T,4,4] probe->world; output relative-next-probe targets."""
    rel = torch.linalg.inv(poses[:, 1:]) @ poses[:, :-1]
    trans = rel[..., :3, 3]
    if task == 'lateral':
        return trans[..., 0:1]
    if task == 'inplane_rot':
        angle = torch.atan2(rel[..., 1, 0], rel[..., 0, 0])
        return angle.unsqueeze(-1)
    if task == 'elev_mag':
        return trans[..., 2:3].abs()
    if task == 'elev_signed':
        return trans[..., 2:3]
    # For full 6-DoF evaluate in the small-rotation regime: use exact SO(3) log.
    # Implement matrix->rotvec on CPU here only for synthetic GT, no gradients needed.
    r = rel[..., :3, :3].detach().cpu().numpy()
    rv = Rotation.from_matrix(r.reshape(-1, 3, 3)).as_rotvec().reshape(*r.shape[:-2], 3)
    rotvec = torch.from_numpy(rv.copy()).to(trans)
    if task == 'rot_x':
        return rotvec[..., 0:1]
    if task == 'rot_y':
        return rotvec[..., 1:2]
    return torch.cat([rotvec, trans], dim=-1)


def phantom_batch(batch_size, frames=5, image_size=(96, 96), task='lateral',
                  spacing_mm=0.4, device='cpu', generator=None,
                  signed_speed_px=None, shared_volume=False):
    """A stationary random 3-D scattering field sampled with controlled probe poses.

    This is a unit-test / feasibility phantom, NOT an acoustically realistic
    alternative to SonoGym. 3-D trilinear sampling preserves cross-frame texture
    coherence and differentiable subpixel motion. No motion metadata enters images.
    """
    if frames < 2:
        raise ValueError('frames must be >=2')
    if task not in ('lateral', 'inplane_rot', 'rot_x', 'rot_y',
                    'elev_mag', 'elev_signed', 'rigid'):
        raise ValueError(task)
    h, w = image_size
    b, t = batch_size, frames
    rand = lambda *shape: torch.rand(shape, device=device, generator=generator)
    depth, margin = 48, 32
    height, width = h + 2 * margin, w + 2 * margin
    noise = rand(b, 1, depth, height, width)
    noise = F.avg_pool3d(noise, 3, stride=1, padding=1)
    noise = F.avg_pool3d(noise, 3, stride=1, padding=1)
    # Local anatomical texture: add a smooth, spatially varying background.
    volume = (noise - noise.mean(dim=(2, 3, 4), keepdim=True)) / noise.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-5)
    volume = (volume * 0.20 + 0.5).clamp(0, 1)
    if shared_volume:
        volume = volume[:1].expand(b, -1, -1, -1, -1).contiguous()
    motion = torch.zeros(b, 3, device=device)
    angles = torch.zeros(b, 3, device=device)
    if task == 'lateral':
        motion[:, 0] = ((rand(b) * 2 - 1) * 2.0 if signed_speed_px is None else
                        torch.as_tensor(signed_speed_px, device=device, dtype=torch.float32))
        if motion[:, 0].shape != (b,):
            raise ValueError('signed_speed_px must provide one speed per video')
    elif task in ('elev_mag', 'elev_signed'):
        motion[:, 2] = (rand(b) * 2 - 1) * 1.8
    elif task == 'inplane_rot':
        angles[:, 2] = (rand(b) * 2 - 1) * 0.045
    elif task == 'rot_x':
        angles[:, 0] = (rand(b) * 2 - 1) * 0.045
    elif task == 'rot_y':
        angles[:, 1] = (rand(b) * 2 - 1) * 0.045
    else:
        motion = (rand(b, 3) * 2 - 1) * torch.tensor([1.6, 1.2, 1.6], device=device)
        angles = (rand(b, 3) * 2 - 1) * 0.025
    times = torch.arange(t, device=device).float()[None, :, None]
    translation_px = times * motion[:, None]
    angles_t = times * angles[:, None]
    rotation = _rotations(angles_t)
    poses = torch.eye(4, device=device).expand(b, t, 4, 4).clone()
    poses[..., :3, :3] = rotation
    poses[..., :3, 3] = translation_px * spacing_mm
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    xy = torch.stack([xx.float() - (w - 1) / 2, yy.float() - (h - 1) / 2, torch.zeros_like(xx).float()], -1)
    world = torch.einsum('btij,hwj->bthwi', rotation, xy) + translation_px[:, :, None, None, :]
    ix = 2 * (world[..., 0] + (width - 1) / 2) / (width - 1) - 1
    iy = 2 * (world[..., 1] + (height - 1) / 2) / (height - 1) - 1
    iz = 2 * (world[..., 2] + (depth - 1) / 2) / (depth - 1) - 1
    grid = torch.stack([ix, iy, iz], -1)
    video = F.grid_sample(volume, grid, mode='bilinear', padding_mode='border', align_corners=True)
    video = video.permute(0, 2, 1, 3, 4).contiguous()
    # Known physical pose target, independent of texture / RNG.
    target = _labels_from_poses(poses, task)
    return video, target.to(video), poses


class SweepWindows(Dataset):
    """Read CT2Eco generated sweeps; patient/case-disjoint existing split.

    Example: SweepWindows('data/pilot_v2', 'train', task='rigid', frames=5)
    Index selects a reproducible random sweep and contiguous window. We sample
    only adjacent transformations to isolate motion perception from frame-gap priors.
    """
    def __init__(self, root, split, task='rigid', frames=5, length=16000, seed=17,
                 expected_motion=None, return_spacing=False):
        self.root, self.task, self.frames, self.length, self.seed = Path(root), task, frames, length, seed
        self.return_spacing = return_spacing
        split_path = self.root / 'metadata' / 'splits.json'
        allowed = set(json.loads(split_path.read_text())[split])
        self.entries = [p for p in sorted((self.root / 'sweeps').iterdir())
                        if p.is_dir() and (p / 'sweep.json').exists()
                        and json.loads((p / 'sweep.json').read_text())['case_id'] in allowed]
        if not self.entries:
            raise ValueError(f'No sweeps in {split} at {root}')
        if expected_motion is not None:
            # sweeps built before the 'motion' field existed were all random-motion
            motions = [(e.name, json.loads((e / 'sweep.json').read_text()).get('motion', 'random'))
                       for e in self.entries]
            unexpected = [m for m in motions if m[1] != expected_motion]
            if unexpected:
                raise ValueError(f'Expected pure motion={expected_motion!r}, found {unexpected[:3]} '
                                 '(generate a dedicated dataset, not mixed random sweeps)')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index)
        entry = self.entries[int(rng.integers(len(self.entries)))]
        x = np.load(entry / 'frames.npy', mmap_mode='r')
        p = np.load(entry / 'poses.npy', mmap_mode='r')
        if len(x) < self.frames:
            raise ValueError(f'Too short sweep: {entry}')
        start = int(rng.integers(len(x) - self.frames + 1))
        clip = torch.from_numpy(np.array(x[start:start + self.frames], dtype=np.float32, copy=True))[:, None] / 255
        poses = torch.from_numpy(np.array(p[start:start + self.frames], dtype=np.float32, copy=True))
        target = _labels_from_poses(poses[None], self.task)[0]
        if self.return_spacing:
            meta = json.loads((entry / 'sweep.json').read_text())
            return clip, target, poses, torch.tensor(meta['spacing_mm'], dtype=torch.float32)
        return clip, target, poses
