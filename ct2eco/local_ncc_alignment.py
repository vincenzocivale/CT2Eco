"""Pose-oracle and image-only in-plane compensation for local B-mode NCC.

Diagnostic only. Oracle geometry is NEVER an encoder input; it is used here to
diagnose ideal geometric registration from source to target coordinates.
Image-only offsets are computed by coarse local NCC matching, without poses.
"""

import math

import torch
from torch.nn import functional as F

from .local_ncc import spatial_ncc


def oracle_flow(poses, spacing_mm, image_hw):
    """Source-to-target image displacement (du,dv) in *pixels*, and |dz| in mm.

    poses: [B,2,4,4] probe-to-world. spacing_mm: [B,2] (dy,dx).
    The relative transform is inv(T_target) @ T_source. The returned flow is
    the orthogonal in-plane projection even where tissue leaves the beam.
    """
    if poses.ndim != 4 or poses.shape[1:] != (2, 4, 4):
        raise ValueError('poses must be [B,2,4,4]')
    if spacing_mm.shape != (len(poses), 2):
        raise ValueError('spacing_mm must be [B,2]')
    if not (spacing_mm > 0).all():
        raise ValueError('Pixel spacings must be positive')
    h, w = image_hw
    rel = torch.linalg.inv(poses[:, 1]) @ poses[:, 0]
    yy, xx = torch.meshgrid(torch.arange(h, device=poses.device, dtype=poses.dtype),
                            torch.arange(w, device=poses.device, dtype=poses.dtype), indexing='ij')
    x = (xx[None] - (w - 1) / 2) * spacing_mm[:, 1, None, None]
    y = yy[None] * spacing_mm[:, 0, None, None]
    rx, ry, rz = [rel[:, i] for i in range(3)]
    xp = rx[:, 0, None, None] * x + rx[:, 1, None, None] * y + rx[:, 3, None, None]
    yp = ry[:, 0, None, None] * x + ry[:, 1, None, None] * y + ry[:, 3, None, None]
    zp = rz[:, 0, None, None] * x + rz[:, 1, None, None] * y + rz[:, 3, None, None]
    du = (xp - x) / spacing_mm[:, 1, None, None]
    dv = (yp - y) / spacing_mm[:, 0, None, None]
    return torch.stack((du, dv), 1), zp.abs()[:, None]


def estimated_flow(source, target, downsample=4, radius=4, patch=7,
                   minimum_score=0.20, shift_penalty=0.015):
    """Estimate source->target flow using ONLY B-mode images and local matching.

    Block matching on a coarse image pyramid level; search over integer offsets,
    score with windowed Pearson NCC, and return a smoothed dense field in native
    pixel units. No pose, spacing, labels, or other sweep data are used.
    This is an inexpensive registration control, not an optimal optical flow.
    Beware winner's curse: maximizing NCC also changes its noise distribution.
    """
    if source.shape != target.shape or source.ndim != 4 or source.shape[1] != 1:
        raise ValueError('source and target must both be [B,1,H,W]')
    if downsample < 1 or radius < 0 or patch < 3 or patch % 2 != 1:
        raise ValueError('Invalid matching parameters')
    h, w = source.shape[-2:]
    fh, fw = math.ceil(h / downsample), math.ceil(w / downsample)
    a = F.interpolate(source.float(), (fh, fw), mode='area')
    b = F.interpolate(target.float(), (fh, fw), mode='area')
    r = radius
    pad = F.pad(b, (r, r, r, r))
    one = F.pad(torch.ones_like(b), (r, r, r, r))
    pool = lambda x: F.avg_pool2d(x, patch, stride=1, padding=patch // 2,
                                  count_include_pad=False)
    am = pool(a)
    av = (pool(a * a) - am.square()).clamp_min(0)
    best = torch.full_like(am, -torch.inf)
    ud = torch.zeros_like(am)
    vd = torch.zeros_like(am)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            z = pad[..., r + dy:r + dy + fh, r + dx:r + dx + fw]
            coverage = one[..., r + dy:r + dy + fh, r + dx:r + dx + fw]
            zm = pool(z)
            zv = (pool(z * z) - zm.square()).clamp_min(0)
            corr = (pool(a * z) - am * zm) / (av * zv).clamp_min(1e-12).sqrt()
            valid = (av > 1e-5) & (zv > 1e-5) & (pool(coverage) > .95)
            score = corr - shift_penalty * (dx * dx + dy * dy) / max(r * r, 1)
            score = score.masked_fill(~valid, -torch.inf)
            better = score > best
            best = torch.where(better, score, best)
            ud = torch.where(better, dx, ud)
            vd = torch.where(better, dy, vd)
    usable = torch.isfinite(best) & (best >= minimum_score)
    # Use local confidence when smoothing, rather than filling invalid zones
    # with arbitrary offsets. Border regions without matches fall back to zero.
    confidence = usable.float()
    denom = F.avg_pool2d(confidence, 3, 1, 1).clamp_min(1e-6)
    ud = F.avg_pool2d(ud * confidence, 3, 1, 1) / denom
    vd = F.avg_pool2d(vd * confidence, 3, 1, 1) / denom
    flow = torch.cat((ud, vd), 1)
    flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)
    flow[:, 0] *= w / fw
    flow[:, 1] *= h / fh
    confidence = F.interpolate(confidence, size=(h, w), mode='nearest')
    return flow, confidence


def warp_target(target, flow):
    """Sample target frame at (source coordinates + source->target flow)."""
    if target.ndim != 4 or flow.shape != (len(target), 2, *target.shape[-2:]):
        raise ValueError('Bad target or flow shape')
    n, _, h, w = target.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=target.device, dtype=flow.dtype),
                            torch.arange(w, device=target.device, dtype=flow.dtype), indexing='ij')
    u = xx[None] + flow[:, 0]
    v = yy[None] + flow[:, 1]
    inside = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    grid = torch.stack((2 * u / max(w - 1, 1) - 1,
                        2 * v / max(h - 1, 1) - 1), -1)
    warped = F.grid_sample(target, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return warped, inside[:, None]


def aligned_ncc(source, target, flow, windows=(15, 31), out_hw=None,
                min_overlap=0.95):
    """NCC on source and warped target, with a geometrical window-validity mask."""
    warped, inside = warp_target(target, flow)
    video = torch.stack((source, warped), dim=1)
    ncc, valid = spatial_ncc(video, windows=windows, out_hw=out_hw)
    if out_hw is None:
        out_hw = ncc.shape[-2:]
    for kidx, k in enumerate(windows):
        cover = F.avg_pool2d(inside.float(), k, 1, k // 2, count_include_pad=False)
        cover = F.interpolate(cover, size=out_hw, mode='area')
        valid[:, 0, kidx] &= cover[:, 0] >= min_overlap
    return ncc[:, 0], valid[:, 0]


def three_way_ncc(video, poses, spacing, windows=(15, 31),
                  downsample=4, radius=4, patch=7):
    """Three NCC conditions for matched paired evaluation. All maps [B,K,h,w]."""
    if video.ndim != 5 or video.shape[1:3] != (2, 1):
        raise ValueError('video must be [B,2,1,H,W]')
    source, target = video[:, 0], video[:, 1]
    h, w = source.shape[-2:]
    fixed, fixed_valid = spatial_ncc(video, windows)
    fixed, fixed_valid = fixed[:, 0], fixed_valid[:, 0]
    oracle, abs_z = oracle_flow(poses, spacing, (h, w))
    est, confidence = estimated_flow(source, target, downsample, radius, patch)
    oracle_ncc, oracle_valid = aligned_ncc(source, target, oracle, windows, fixed.shape[-2:])
    est_ncc, est_valid = aligned_ncc(source, target, est, windows, fixed.shape[-2:])
    # A label-derived in-FOV mask applies to ALL conditions, only for fair
    # comparison and analysis. It is not an input of the estimated alignment.
    _, in_fov = warp_target(target, oracle)
    oracle_overlap = F.interpolate(in_fov.float(), fixed.shape[-2:], mode='area')[:, 0] >= .95
    common = fixed_valid.all(1) & oracle_valid.all(1) & est_valid.all(1) & oracle_overlap
    dz = F.interpolate(abs_z, fixed.shape[-2:], mode='area')[:, 0]
    xy_mm = torch.sqrt((oracle[:, 0] * spacing[:, 1, None, None]).square()
                       + (oracle[:, 1] * spacing[:, 0, None, None]).square())
    xy_mm = F.interpolate(xy_mm[:, None], fixed.shape[-2:], mode='area')[:, 0]
    flow_epe = torch.linalg.vector_norm(est - oracle, dim=1)
    flow_epe = F.interpolate(flow_epe[:, None], fixed.shape[-2:], mode='area')[:, 0]
    return {'ncc': {'fixed': fixed, 'oracle': oracle_ncc, 'estimated': est_ncc},
            'valid': common, 'dz': dz, 'xy_mm': xy_mm, 'flow_epe_px': flow_epe,
            'estimated_confidence': F.interpolate(confidence, fixed.shape[-2:], mode='area')[:, 0]}
