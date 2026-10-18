"""T1: point-query readout from *frozen* L1 representations.

Only the query, frame gap and frozen video features enter the readout.
GT poses are forbidden in this module (they may only construct labels outside).
Q0/Q1/Q2 use exactly the same parameterized decoder and output conventions.
"""

import torch
from torch import nn
from torch.nn import functional as F


BANDS = ('fov_all', 'proxy_overlap', 'z_0_0.5', 'z_0.5_1.5', 'z_1.5_3', 'z_3_inf')
GAPS = (1, 2, 4)


def sample_map(features, queries):
    """Bilinear query samples from a [B,C,h,w] feature map at [Q,2] cell coords."""
    if features.ndim != 4 or queries.ndim != 2 or queries.shape[-1] != 2:
        raise ValueError('Need features [B,C,h,w], queries [Q,2]')
    b, c, h, w = features.shape
    q = queries.to(features).float()
    grid = torch.stack((2 * q[:, 0] / max(w - 1, 1) - 1,
                        2 * q[:, 1] / max(h - 1, 1) - 1), -1)
    sampled = F.grid_sample(features, grid[None, None].expand(b, -1, -1, -1),
                            mode='bilinear', padding_mode='border', align_corners=True)
    return sampled[:, :, 0].transpose(1, 2)  # [B,Q,C]


class PointQueryReadout(nn.Module):
    """Identical lightweight query head for no-image/random/pretrained controls.

    Input: query position (source feature cells), target frame index, and
    optional frozen L1 frame/transition feature maps. Output: target native UV
    pixels plus nonnegative elevational distance mm, in [B,Q,3]. No poses or
    labels can be supplied to forward().
    """
    def __init__(self, channels=128, hidden=192):
        super().__init__()
        self.channels = channels
        # source frame, target frame, mean local transitions, mean global transitions
        self.net = nn.Sequential(nn.Linear(channels * 4 + 3, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 3))
        # Identity XY and near-zero elevation at init, shared between arms.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, queries, gap, image_hw, batch_size, frozen=None):
        if gap not in GAPS:
            raise ValueError('T1 only supports gap 1,2,4')
        if queries.ndim != 2 or queries.shape[1] != 2:
            raise ValueError('Queries must be [Q,2] feature coordinates')
        device = queries.device
        if frozen is not None:
            frames, transitions = frozen['frame_dense'], frozen['dense']
            if frames is None or frames.ndim != 5 or transitions.ndim != 5:
                raise ValueError('L1 needs per-frame and transition dense feature maps')
            if frames.shape[0] != batch_size or frames.shape[1] < gap + 1:
                raise ValueError('Frozen feature batch/clip mismatch')
            if frames.shape[2] != self.channels or transitions.shape[2] != self.channels:
                raise ValueError('Feature channels do not match decoder')
            h, w = frames.shape[-2:]
            source = sample_map(frames[:, 0], queries)
            target = sample_map(frames[:, gap], queries)
            local = sample_map(transitions[:, :gap].mean(1), queries)
            global_motion = transitions[:, :gap].mean(dim=(1, 3, 4))
            global_motion = global_motion[:, None, :].expand(-1, len(queries), -1)
            features = (source, target, local, global_motion)
        else:
            h, w = (int((image_hw[0] + 3) // 4), int((image_hw[1] + 3) // 4))
            zeros = torch.zeros(batch_size, len(queries), self.channels, device=device)
            features = (zeros, zeros, zeros, zeros)
        if (queries[:, 0] < 0).any() or (queries[:, 0] >= w).any() or (queries[:, 1] < 0).any() or (queries[:, 1] >= h).any():
            raise ValueError('Query outside feature grid')
        uv = queries.float() * 4
        normalized = torch.stack((2 * queries[:, 0] / max(w-1, 1) - 1,
                                  2 * queries[:, 1] / max(h-1, 1) - 1,
                                  torch.full_like(queries[:, 0], gap / 4).float()), -1)
        coords = normalized[None].expand(batch_size, -1, -1)
        hidden = torch.cat([coords, *features], dim=-1)
        raw = self.net(hidden)
        target_uv = uv[None] + raw[..., :2] * 4
        positive_z = F.softplus(raw[..., 2] - 3.)
        return torch.cat([target_uv, positive_z[..., None]], dim=-1)


def loss_point_queries(pred, target_uv, target_abs_z, in_fov, z_weight=1.):
    """Balanced target-only geometric loss, all in-FOV points, no z-based exclusion."""
    if pred.shape != (*target_uv.shape[:-1], 3) or target_abs_z.shape != target_uv.shape[:-1]:
        raise ValueError('Invalid query prediction/target shape')
    if not bool(in_fov.any()):
        raise ValueError('Training batch has no in-FOV query projections')
    image_error = F.smooth_l1_loss(pred[..., :2] / 4, target_uv / 4,
                                  beta=1., reduction='none').sum(-1)
    z_error = F.smooth_l1_loss(pred[..., 2], target_abs_z, beta=.5, reduction='none')
    return (image_error + z_weight * z_error)[in_fov].mean()


def aggregate_query_errors(acc, pred, target_uv, target_abs_z, in_fov, overlap_mm=0.5):
    """Accumulate sums/counts; observations are points, not independent CTs."""
    epe = torch.linalg.vector_norm(pred[..., :2] - target_uv, dim=-1)
    dz_err = (pred[..., 2] - target_abs_z).abs()
    mask_by_band = {'fov_all': in_fov,
                    'proxy_overlap': in_fov & (target_abs_z <= overlap_mm),
                    'z_0_0.5': in_fov & (target_abs_z < .5),
                    'z_0.5_1.5': in_fov & (target_abs_z >= .5) & (target_abs_z < 1.5),
                    'z_1.5_3': in_fov & (target_abs_z >= 1.5) & (target_abs_z < 3),
                    'z_3_inf': in_fov & (target_abs_z >= 3)}
    for band, mask in mask_by_band.items():
        n = int(mask.sum())
        row = acc.setdefault(band, {'n': 0, 'epe_sum': 0., 'dz_sum': 0., 'r4': 0, 'r8': 0})
        row['n'] += n
        if n:
            err = epe[mask]
            row['epe_sum'] += float(err.double().sum())
            row['dz_sum'] += float(dz_err[mask].double().sum())
            row['r4'] += int((err <= 4).sum())
            row['r8'] += int((err <= 8).sum())
    return acc


def finalize_query_errors(acc):
    return {band: {'n_points': stat['n'],
                   'epe_px': stat['epe_sum'] / stat['n'],
                   'abs_dz_mae_mm': stat['dz_sum'] / stat['n'],
                   'recall_at_4px': stat['r4'] / stat['n'],
                   'recall_at_8px': stat['r8'] / stat['n']}
            for band, stat in acc.items() if stat['n'] > 0}
