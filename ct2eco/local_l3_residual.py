"""Frozen-L1 dense elevation refinement, ablation R0/R1/R2.

Only video and physical pixel spacing reach forward(). The 6-DoF pose labels
are never inputs to the encoder, predicted flow, NCC or residual head.

R0: L1 feature map + no NCC evidence (capacity control)
R1: L1 feature map + fixed-grid NCC (two image-window scales)
R2: L1 feature map + fixed NCC + L1-flow-aligned NCC

For exact capacity matching, all variants use one architecture with eight aux
slots: two fixed decorrelations + validity, two aligned decorrelations +
validity. Unused slots are zeros, and the residual output is zero-initialized.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .local_l1_alignment import upsample_probe_mm
from .local_ncc import spatial_ncc
from .local_ncc_alignment import aligned_ncc


class DzResidual(nn.Module):
    """Same parameter count for R0/R1/R2; predicts an additive softplus-logit delta."""

    def __init__(self, channels=128, aux_channels=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels + aux_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, dense, aux):
        if dense.ndim != 5 or aux.ndim != 5 or dense.shape[:2] != aux.shape[:2]:
            raise ValueError('Expected [B,T-1,C,h,w] dense features and auxiliary maps')
        if dense.shape[-2:] != aux.shape[-2:] or aux.shape[2] != 8:
            raise ValueError('Auxiliary maps must have 8 channels on the dense feature grid')
        b, t = dense.shape[:2]
        return self.net(torch.cat((dense, aux), dim=2).flatten(0, 1)).unflatten(0, (b, t))


def refine_positive(base_z_mm, logit_delta):
    """Zero residual gives the frozen positive baseline up to a 1e-5 floor."""
    if base_z_mm.shape != logit_delta.shape:
        raise ValueError('Base z and residual must share dimensions')
    base = base_z_mm.clamp_min(1e-5)
    # Stable inverse of softplus, then a learned change in logit space.
    inverse = base + torch.log(-torch.expm1(-base))
    return F.softplus(inverse + logit_delta)


class LocalL3Residual(nn.Module):
    """R0/R1/R2: frozen L1 teacher + trainable dz-only head."""

    def __init__(self, encoder, probe, variant='r2', windows=(15, 31)):
        super().__init__()
        if variant not in ('r0', 'r1', 'r2'):
            raise ValueError('variant must be r0/r1/r2')
        if len(windows) != 2 or any(k < 3 or k % 2 != 1 for k in windows):
            raise ValueError('Exactly two odd NCC windows >=3 are required')
        self.encoder = encoder
        self.probe = probe
        self.variant = variant
        self.windows = tuple(windows)
        self.refiner = DzResidual()
        for mod in (self.encoder, self.probe):
            for p in mod.parameters():
                p.requires_grad_(False)
            mod.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.probe.eval()
        return self

    def forward(self, video, spacing_mm):
        """No GT poses. Returns L1 baseline, corrected xyz, and fixed NCC mask.

        spacing_mm [B,2] = (dy,dx) in mm/pixel, used only to turn predicted
        in-plane displacement from mm into sampling coordinates in pixels.
        """
        if video.ndim != 5 or video.shape[1] != 5 or video.shape[2] != 1:
            raise ValueError('Frozen L1 requires five-frame ultrasound clips')
        b, t, _, h, w = video.shape
        if spacing_mm.shape != (b, 2) or not (spacing_mm > 0).all():
            raise ValueError('spacing_mm must be positive [B,2] in (dy,dx) order')
        with torch.no_grad():
            dense = self.encoder(video)['dense']
            n = t - 1
            fh, fw = dense.shape[-2:]
            base = self.probe(dense)
            if base.shape != (b, n, 3, fh, fw):
                raise ValueError(f'Unexpected L1 xyz probe shape: {base.shape}')
            fixed_ncc, fixed_valid = spatial_ncc(video, self.windows, (fh, fw))
            aux = dense.new_zeros(b, n, 8, fh, fw)
            if self.variant in ('r1', 'r2'):
                aux[:, :, 0:2] = (1 - fixed_ncc) * fixed_valid.float() * 10
                aux[:, :, 2:4] = fixed_valid.float()
            if self.variant == 'r2':
                # Poses are never used here. Flow is from the frozen L1 head.
                dense_xy = base[:, :, :2].reshape(b * n, 2, fh, fw)
                raw_xy = upsample_probe_mm(dense_xy, (h, w))
                repeated_spacing = spacing_mm.repeat_interleave(n, dim=0)
                flow = raw_xy.clone()
                flow[:, 0] /= repeated_spacing[:, 1, None, None]
                flow[:, 1] /= repeated_spacing[:, 0, None, None]
                a = video[:, :-1].reshape(b * n, 1, h, w)
                z = video[:, 1:].reshape(b * n, 1, h, w)
                aligned, valid = aligned_ncc(a, z, flow, self.windows, (fh, fw))
                aligned = aligned.reshape(b, n, 2, fh, fw)
                valid = valid.reshape(b, n, 2, fh, fw)
                aux[:, :, 4:6] = (1 - aligned) * valid.float() * 10
                aux[:, :, 6:8] = valid.float()
        delta = self.refiner(dense.detach(), aux.detach())
        updated_z = refine_positive(base[:, :, 2:3].detach(), delta)
        prediction = torch.cat((base[:, :, :2].detach(), updated_z), dim=2)
        return {'prediction': prediction, 'baseline': base.detach(),
                'fixed_valid': fixed_valid.detach(), 'residual': delta,
                'ncc_aux': aux.detach()}
