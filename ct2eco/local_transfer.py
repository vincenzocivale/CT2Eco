"""Validation-only, no-training sparse correspondence tests for frozen LocalStream.

Image-only prediction methods never receive poses. Label construction uses GT
poses solely for evaluating the geometric source->target projection. Out-of-plane
projections are *not* automatically visible physical correspondences.
"""
import torch
from torch.nn import functional as F


def query_positions(feature_hw, n_queries=64, border=3, device='cpu'):
    """Fixed query positions in feature cells, independent of ground truth."""
    h, w = feature_hw
    if h <= 2 * border or w <= 2 * border or n_queries < 1:
        raise ValueError('Feature grid too small or invalid query count')
    ny = max(1, int(round((n_queries * (h - 2 * border) / (w - 2 * border)) ** 0.5)))
    nx = max(1, n_queries // ny)
    yy = torch.linspace(border, h - border - 1, ny, device=device).round().long().unique()
    xx = torch.linspace(border, w - border - 1, nx, device=device).round().long().unique()
    y, x = torch.meshgrid(yy, xx, indexing='ij')
    return torch.stack((x.flatten(), y.flatten()), dim=-1)[:n_queries]


def project_queries(poses, spacing_mm, queries, image_hw, source_t=0, target_t=1):
    """Pose-label-only sparse point GT; return target UV px and |elevation| mm.

    Source feature (x,y) cells are anchored at native image pixels (4x,4y).
    Probe axes: x=(u-(W-1)/2)*dx; y=v*dy; source z=0.
    """
    if poses.ndim != 4 or poses.shape[-2:] != (4, 4):
        raise ValueError('poses must have shape [B,T,4,4]')
    if spacing_mm.shape != (len(poses), 2):
        raise ValueError('spacing_mm must have shape [B,2] ordered (dy,dx)')
    if target_t <= source_t or target_t >= poses.shape[1]:
        raise ValueError('Invalid source-target frame indices')
    if (spacing_mm <= 0).any():
        raise ValueError('Pixel spacing must be positive')
    h, w = image_hw
    uv = queries.to(device=poses.device, dtype=poses.dtype) * 4
    u, v = uv.unbind(-1)
    sx = (u[None] - (w - 1) / 2) * spacing_mm[:, 1, None]
    sy = v[None] * spacing_mm[:, 0, None]
    zeros = torch.zeros_like(sx)
    xyz = torch.stack((sx, sy, zeros), dim=-1)
    rel = torch.linalg.inv(poses[:, target_t]) @ poses[:, source_t]
    dest = torch.einsum('bij,bqj->bqi', rel[:, :3, :3], xyz) + rel[:, None, :3, 3]
    target_uv = torch.stack((dest[..., 0] / spacing_mm[:, None, 1] + (w - 1) / 2,
                             dest[..., 1] / spacing_mm[:, None, 0]), dim=-1)
    in_fov = ((target_uv[..., 0] >= 0) & (target_uv[..., 0] <= w - 1) &
              (target_uv[..., 1] >= 0) & (target_uv[..., 1] <= h - 1))
    return target_uv, dest[..., 2].abs(), in_fov


def _normalized(desc, centered=False):
    if centered:
        desc = desc - desc.mean(dim=2, keepdim=True)
    return F.normalize(desc.float(), p=2, dim=2, eps=1e-6)


def raw_patch_descriptors(video, feature_hw, patch=9):
    """Local raw B-mode patches anchored to 4-pixel feature cells."""
    if patch % 2 != 1 or patch < 3:
        raise ValueError('patch must be odd and >=3')
    b, t, c, h, w = video.shape
    if c != 1:
        raise ValueError('Expected B-mode [B,T,1,H,W]')
    desc = F.unfold(video.reshape(b * t, 1, h, w), kernel_size=patch,
                    stride=4, padding=patch // 2)
    fh, fw = feature_hw
    if desc.shape[-1] != fh * fw:
        raise ValueError('Mismatch between raw patch anchors and feature grid')
    return desc.reshape(b, t, patch * patch, fh, fw)


def match_descriptors(source, target, queries, radius=8, centered=False):
    """Feature-space local NN retrieval; NO GT or trained retrieval head.

    Candidate search is query-centered, with fixed radius (not GT-centered).
    Coordinates are returned in native pixels at 4-pixel feature anchors.
    """
    if source.shape != target.shape or source.ndim != 4:
        raise ValueError('source/target maps must have same [B,C,h,w] shape')
    if radius < 0:
        raise ValueError('radius must be nonnegative')
    b, c, h, w = source.shape
    q = queries.long().to(source.device)
    if q.ndim != 2 or q.shape[1] != 2 or ((q[:, 0] < 0) | (q[:, 0] >= w) |
                                                (q[:, 1] < 0) | (q[:, 1] >= h)).any():
        raise ValueError('Queries outside feature grid')
    offsets = torch.arange(-radius, radius + 1, device=source.device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing='ij')
    shift = torch.stack((ox.flatten(), oy.flatten()), dim=-1)
    candidates = q[:, None, :] + shift[None, :, :]
    legal = ((candidates[..., 0] >= 0) & (candidates[..., 0] < w) &
             (candidates[..., 1] >= 0) & (candidates[..., 1] < h))
    target_idx = candidates[..., 1].clamp(0, h-1) * w + candidates[..., 0].clamp(0, w-1)
    source_idx = q[:, 1] * w + q[:, 0]
    src = source.flatten(2)[:, :, source_idx].transpose(1, 2)  # B,Q,C
    tgt = target.flatten(2)[:, :, target_idx.flatten()].reshape(b, c, len(q), -1)
    tgt = tgt.permute(0, 2, 3, 1)  # B,Q,K,C
    if centered:
        src = src - src.mean(dim=-1, keepdim=True)
        tgt = tgt - tgt.mean(dim=-1, keepdim=True)
    src = F.normalize(src.float(), dim=-1, eps=1e-6)
    tgt = F.normalize(tgt.float(), dim=-1, eps=1e-6)
    similarity = (src[:, :, None, :] * tgt).sum(-1).masked_fill(~legal[None], -torch.inf)
    best = similarity.argmax(-1)
    estimated_xy = candidates[torch.arange(len(q), device=q.device)[None].expand(b,-1), best]
    return estimated_xy.float() * 4, similarity.gather(-1, best[..., None]).squeeze(-1)


def sample_dense_motion(dense_xy_mm, feature_uv, spacing_mm):
    """Bilinearly sample L1 next-step (dx,dy) mm at arbitrary feature coords."""
    b, c, h, w = dense_xy_mm.shape
    if c != 2 or feature_uv.shape[:1] != (b,) or feature_uv.shape[-1] != 2:
        raise ValueError('Wrong flow/map/query shape')
    norm = torch.stack((2 * feature_uv[..., 0] / max(w - 1, 1) - 1,
                        2 * feature_uv[..., 1] / max(h - 1, 1) - 1), dim=-1)
    z = F.grid_sample(dense_xy_mm, norm[:, None], mode='bilinear',
                      padding_mode='border', align_corners=True)[:, :, 0].transpose(1, 2)
    return z / torch.stack((spacing_mm[:, 1], spacing_mm[:, 0]), dim=-1)[:, None]


def compose_l1_displacements(xyz_pred, queries, spacing_mm, gap):
    """Compose source->target 2D projected motions across adjacent transitions.

    This is a trained *head baseline* rather than an independent new task probe.
    Out-of-plane nonvisibility is not addressed by 2D composition.
    """
    if xyz_pred.ndim != 5 or xyz_pred.shape[2] < 2 or not 1 <= gap <= xyz_pred.shape[1]:
        raise ValueError('Expected B,T-1,C,h,w and valid temporal gap')
    coords = queries.to(xyz_pred.device).float()[None].expand(len(xyz_pred), -1, -1).clone()
    for t in range(gap):
        coords = coords + sample_dense_motion(xyz_pred[:, t, :2], coords, spacing_mm) / 4
    return coords * 4


def query_metrics(target_uv, dz, fov, estimates, max_dz_mm=0.5, search_radius=None):
    """Report all in-FOV projections and distance-stratified proxy-visible subset."""
    bands = {'proxy_overlap': (dz <= max_dz_mm) & fov,
             'fov_all': fov,
             'z_0_0.5': (dz < 0.5) & fov,
             'z_0.5_1.5': (dz >= 0.5) & (dz < 1.5) & fov,
             'z_1.5_3': (dz >= 1.5) & (dz < 3) & fov,
             'z_3_inf': (dz >= 3) & fov}
    out = {}
    for key, mask in bands.items():
        n = int(mask.sum())
        out[key] = {'n': n}
        if search_radius is not None and n:
            # Geometric diagnostic only: does the true projection lie inside
            # the query-centered square search window? Do not mask it out.
            offset = (target_uv - estimates['identity']).abs()
            out[key]['search_reachable'] = int(((offset[..., 0] <= search_radius * 4) &
                                                 (offset[..., 1] <= search_radius * 4) & mask).sum())
        if n:
            for name, xy in estimates.items():
                epe = torch.linalg.vector_norm(xy - target_uv, dim=-1)[mask]
                out[key][name] = {'epe_sum_px': float(epe.double().sum()),
                                  'within_4px': int((epe <= 4).sum()),
                                  'within_8px': int((epe <= 8).sum())}
    return out
