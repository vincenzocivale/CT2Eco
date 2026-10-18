"""Dense multiscale normalized cross-correlation on B-mode frame pairs.

Pure PyTorch, no learned weights. Intentionally *unregistered*: uses image
intensities only, never CT anatomy, trajectories, GT labels, case IDs or poses.
The initial L2b experiment uses zero spatial search (appropriate to pure
out-of-plane probes); in-plane movement must be registered before NCC is
interpreted as an elevational distance.
"""

import math

import torch
from torch.nn import functional as F


def spatial_ncc(video, windows=(15, 31), out_hw=None, min_variance=1e-5, eps=1e-6):
    """Return (NCC maps, validity masks), shape [B,T-1,K,h,w].

    video: [B,T,1,H,W], intensities in [0,1]. Each location is a *local*
    Pearson NCC for adjacent frames over a raw-image square window, followed
    by area aggregation to the dense encoder resolution. Boundary windows use
    count_include_pad=False, so artificial black padding isn't considered.
    """
    if video.ndim != 5 or video.shape[1] < 2 or video.shape[2] != 1:
        raise ValueError('Expected [B,T>=2,1,H,W] video')
    if not windows or any(k < 3 or k % 2 != 1 for k in windows):
        raise ValueError('NCC windows must be positive odd sizes >=3')
    b, t, _, h, w = video.shape
    if out_hw is None:
        out_hw = (math.ceil(h / 4), math.ceil(w / 4))
    if out_hw[0] < 1 or out_hw[1] < 1:
        raise ValueError('Invalid NCC output resolution')
    # float32 avoids catastrophic cancellation of image-moment differences
    # with mixed precision or low-variance homogeneous B-mode backgrounds.
    a = video[:, :-1].reshape(-1, 1, h, w).float()
    z = video[:, 1:].reshape(-1, 1, h, w).float()
    values, masks = [], []
    for k in windows:
        pool = lambda q: F.avg_pool2d(q, kernel_size=k, stride=1, padding=k // 2,
                                      count_include_pad=False)
        mean_a, mean_z = pool(a), pool(z)
        var_a = (pool(a * a) - mean_a.square()).clamp_min(0)
        var_z = (pool(z * z) - mean_z.square()).clamp_min(0)
        covariance = pool(a * z) - mean_a * mean_z
        ncc = covariance / (var_a * var_z).clamp_min(eps * eps).sqrt()
        valid = (var_a >= min_variance) & (var_z >= min_variance)
        ncc = ncc.clamp(-1, 1).masked_fill(~valid, 0)
        # Fixed-size average makes each output cell represent an *area* of
        # native-resolution NCC measurements, not a subsampled pixel.
        down = F.interpolate(ncc, size=out_hw, mode='area')
        coverage = F.interpolate(valid.float(), size=out_hw, mode='area')
        values.append(down.reshape(b, t - 1, *out_hw))
        masks.append((coverage >= 0.75).reshape(b, t - 1, *out_hw))
    return torch.stack(values, dim=2), torch.stack(masks, dim=2)


def dense_abs_z_from_poses(poses, image_hw, spacing_mm, feature_hw):
    """Exact |target-probe z| for each source probe-plane pixel on dense grid.

    poses [B,T,4,4] are probe-to-world; spacing_mm [B,2] is (dy,dx) mm.
    The 1/4-resolution grid follows the convolution centres at pixel indices
    (0,4,8,...), not align_corners=True interpolation centres. No pose may be
    passed to LocalStream.forward; this function is GT construction ONLY.
    """
    if poses.ndim != 4 or poses.shape[-2:] != (4, 4):
        raise ValueError('Expected poses [B,T,4,4]')
    if spacing_mm.shape != (poses.shape[0], 2):
        raise ValueError('Expected spacings [B,2] in (dy,dx) mm')
    h, w = image_hw
    fh, fw = feature_hw
    y = (torch.arange(fh, device=poses.device, dtype=poses.dtype) * 4).clamp(max=h - 1)
    x = (torch.arange(fw, device=poses.device, dtype=poses.dtype) * 4).clamp(max=w - 1)
    xx = (x[None, :] - (w - 1) / 2).expand(fh, fw)
    yy = y[:, None].expand(fh, fw)
    rel = torch.linalg.inv(poses[:, 1:]) @ poses[:, :-1]
    zz = (rel[..., 2, 0, None, None] * xx * spacing_mm[:, None, None, 1, None]
          + rel[..., 2, 1, None, None] * yy * spacing_mm[:, None, None, 0, None]
          + rel[..., 2, 3, None, None])
    return zz.abs().unsqueeze(2)  # [B,T-1,1,fh,fw], mm


def dense_motion_from_poses(poses, image_hw, spacing_mm, feature_hw, return_fov=False):
    """Pose-only ground truth: (dx,dy,|dz|) in mm on the source dense grid.

    Points are expressed in next-frame probe coordinates using
    inv(T_next) @ T_source. No poses, spacings or labels enter the encoder.
    Optional `in_fov` excludes projected correspondences outside the next
    image's rectangular field of view, without excluding out-of-plane points.
    """
    if poses.ndim != 4 or poses.shape[-2:] != (4, 4):
        raise ValueError('Expected poses [B,T,4,4]')
    if spacing_mm.shape != (poses.shape[0], 2):
        raise ValueError('Expected spacings [B,2] in (dy,dx) mm')
    h, w = image_hw
    fh, fw = feature_hw
    y = (torch.arange(fh, device=poses.device, dtype=poses.dtype) * 4).clamp(max=h-1)
    x = (torch.arange(fw, device=poses.device, dtype=poses.dtype) * 4).clamp(max=w-1)
    xx = (x[None, :] - (w - 1) / 2).expand(fh, fw)[None, None] * spacing_mm[:, None, None, 1, None]
    yy = y[:, None].expand(fh, fw)[None, None] * spacing_mm[:, None, None, 0, None]
    rel = torch.linalg.inv(poses[:, 1:]) @ poses[:, :-1]
    new_x = rel[..., 0, 0, None, None] * xx + rel[..., 0, 1, None, None] * yy + rel[..., 0, 3, None, None]
    new_y = rel[..., 1, 0, None, None] * xx + rel[..., 1, 1, None, None] * yy + rel[..., 1, 3, None, None]
    new_z = rel[..., 2, 0, None, None] * xx + rel[..., 2, 1, None, None] * yy + rel[..., 2, 3, None, None]
    target = torch.stack((new_x - xx, new_y - yy, new_z.abs()), dim=2)
    if not return_fov:
        return target
    src_px_u = new_x / spacing_mm[:, None, None, 1, None] + (w - 1) / 2
    src_px_v = new_y / spacing_mm[:, None, None, 0, None]
    overlap = ((src_px_u >= 0) & (src_px_u <= w-1) &
               (src_px_v >= 0) & (src_px_v <= h-1))
    return target, overlap.unsqueeze(2)
