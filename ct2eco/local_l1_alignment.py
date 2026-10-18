"""Use an existing L1 dense-motion checkpoint as image-only in-plane registration.

Diagnostic only. GT poses/spacings may be used for ground-truth evaluation, but
predicted registration receives ONLY the video and probe pixel spacing. Output
flow is source->target in (du,dv) pixels, consistent with warp_target().
"""

from pathlib import Path

import torch
from torch.nn import functional as F

from .local_stream import LocalStream


def load_l1_dense_checkpoint(path, device='cpu'):
    """Restore a previously trained mixed-motion L1 + xyz probe, no training."""
    # Trust only checkpoints created by this project: torch checkpoint loading
    # is not a safe operation on files from unknown sources.
    from scripts.train_local_dense import DenseXYZHead

    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    expected = {'arch': 'l1', 'motion': 'random', 'head': 'xyz'}
    mismatched = {key: (config.get(key), value) for key, value in expected.items()
                  if config.get(key) != value}
    if mismatched:
        raise ValueError(f'Checkpoint must be an L1 random-motion xyz model: {mismatched}')
    encoder = LocalStream('l1', 'rigid', frame_norm=config.get('frame_norm', False)).to(device)
    probe = DenseXYZHead().to(device)
    encoder.load_state_dict(ckpt['encoder'], strict=True)
    probe.load_state_dict(ckpt['dense_probe'], strict=True)
    encoder.eval()
    probe.eval()
    for module in (encoder, probe):
        for p in module.parameters():
            p.requires_grad_(False)
    return encoder, probe


def upsample_probe_mm(dense_xy_mm, image_hw):
    """Map feature cells anchored at raw pixel (4*j,4*i) to the raw grid.

    Dense labels in CT2Eco are constructed at (4*j,4*i), not at the centres of
    conventional resize cells. grid_sample uses the matching coordinate
    convention (and extends the nearest feature cell at image boundaries).
    """
    if dense_xy_mm.ndim != 4 or dense_xy_mm.shape[1] != 2:
        raise ValueError('Need [B,2,h,w] in-plane flow in mm')
    h, w = image_hw
    fh, fw = dense_xy_mm.shape[-2:]
    ys = torch.arange(h, dtype=dense_xy_mm.dtype, device=dense_xy_mm.device) / 4
    xs = torch.arange(w, dtype=dense_xy_mm.dtype, device=dense_xy_mm.device) / 4
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack((2 * xx / max(fw - 1, 1) - 1,
                        2 * yy / max(fh - 1, 1) - 1), -1)
    return F.grid_sample(dense_xy_mm.float(), grid[None].expand(len(dense_xy_mm), -1, -1, -1),
                         mode='bilinear', padding_mode='border', align_corners=True)


@torch.no_grad()
def l1_inplane_flow(encoder, probe, five_frame_video, spacing_mm):
    """Predict pair 0->1, L1 motion [dx,dy] mm -> [du,dv] image pixels.

    five_frame_video: [B,5,1,H,W], pixels [0,1]; probe spacings [B,2]
    ordered as (dy,dx) mm/px. No pose or label arguments are accepted.
    """
    if five_frame_video.ndim != 5 or five_frame_video.shape[1:3] != (5, 1):
        raise ValueError('L1 was trained with five-frame clips [B,5,1,H,W]')
    if spacing_mm.shape != (len(five_frame_video), 2) or not (spacing_mm > 0).all():
        raise ValueError('spacing_mm must have positive (dy,dx) per sample')
    transition = encoder(five_frame_video)['dense'][:, 0]
    dense_xy_mm = probe.net(transition)[:, :2]  # DenseXYZHead: first two unconstrained
    xy_mm = upsample_probe_mm(dense_xy_mm, five_frame_video.shape[-2:])
    flow = xy_mm.clone()
    flow[:, 0] /= spacing_mm[:, 1, None, None]
    flow[:, 1] /= spacing_mm[:, 0, None, None]
    return flow
