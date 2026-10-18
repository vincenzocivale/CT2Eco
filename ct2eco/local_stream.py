"""Local motion encoder experiments for CT2Eco.

L0: early-fusion 2-D CNN applied to adjacent frame pairs.
L1: temporal-stride-1 3-D ResNet, preserving dense per-frame maps.
L2: L1 with an optional local feature-correlation volume.
L2b: L1 with multiscale *image-domain* spatial NCC and residual fusion.

Only frames enter the networks; trajectory / pose annotations are never inputs.
All variants return dense transition maps for later integration with the
D4RT query decoder. The diagnostic head is disposable.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .local_ncc import spatial_ncc


def _norm(channels):
    return nn.GroupNorm(8, channels)


class FrameGroupNorm(nn.GroupNorm):
    """GroupNorm per time step on [B,C,T,H,W]: statistics never mix frames."""
    def forward(self, x):
        if x.ndim != 5:
            return super().forward(x)
        b, c, t, h, w = x.shape
        y = super().forward(x.transpose(1, 2).reshape(b * t, c, h, w))
        return y.reshape(b, t, c, h, w).transpose(1, 2)


def _norm3d(channels, frame_norm):
    return FrameGroupNorm(8, channels) if frame_norm else _norm(channels)


class Residual2D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            _norm(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), _norm(cout),
        )
        self.skip = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), _norm(cout)))

    def forward(self, x):
        return F.gelu(self.body(x) + self.skip(x))


class Residual3D(nn.Module):
    def __init__(self, cin, cout, spatial_stride=1, frame_norm=False):
        super().__init__()
        stride = (1, spatial_stride, spatial_stride)
        norm = lambda c: _norm3d(c, frame_norm)
        self.body = nn.Sequential(
            nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False),
            norm(cout), nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), norm(cout),
        )
        self.skip = (nn.Identity() if cin == cout and spatial_stride == 1 else
                     nn.Sequential(nn.Conv3d(cin, cout, 1, stride=stride, bias=False), norm(cout)))

    def forward(self, x):
        return F.gelu(self.body(x) + self.skip(x))


class Pair2D(nn.Module):
    """L0: early fusion over [I_t, I_(t+1)]. Output [B,T-1,C,h,w]."""
    def __init__(self, channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2, bias=False), _norm(32), nn.GELU(),
            Residual2D(32, 64, stride=2), Residual2D(64, 64),
            Residual2D(64, channels), Residual2D(channels, channels),
        )

    def forward(self, video):
        b, t, _, h, w = video.shape
        pairs = torch.cat([video[:, :-1], video[:, 1:]], dim=2).reshape(b * (t - 1), 2, h, w)
        maps = self.net(pairs)
        return maps.reshape(b, t - 1, *maps.shape[1:])


class Temporal3D(nn.Module):
    """L1: 3-D convolutions with no temporal downsampling; per-frame dense maps.

    frame_norm=True normalizes each time step separately, so features do not
    depend on clip length through normalization statistics.
    """
    def __init__(self, channels=128, frame_norm=False):
        super().__init__()
        fn = frame_norm
        self.net = nn.Sequential(
            nn.Conv3d(1, 32, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            _norm3d(32, fn), nn.GELU(),
            Residual3D(32, 64, 2, fn), Residual3D(64, 64, 1, fn),
            Residual3D(64, channels, 1, fn), Residual3D(channels, channels, 1, fn),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1), _norm(channels), nn.GELU(),
        )

    def forward(self, video):
        b, t = video.shape[:2]
        x = self.net(video.transpose(1, 2))  # B,C,T,h,w
        frames = x.transpose(1, 2)          # B,T,C,h,w
        a, z = frames[:, :-1], frames[:, 1:]
        pairs = torch.cat([a, z, z - a, (z - a).abs()], dim=2)
        f = self.fuse(pairs.flatten(0, 1))
        return f.unflatten(0, (b, t - 1)), frames


class LocalCorrelation(nn.Module):
    """L2: dense normalized local correlation (odd window; e.g. 9x9)."""
    def __init__(self, channels=128, radius=4, corr_channels=32):
        super().__init__()
        self.radius = radius
        self.embedding = nn.Conv2d(channels, corr_channels, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels + (2 * radius + 1) ** 2, channels, 1),
            _norm(channels), nn.GELU(),
        )

    def forward(self, transition, frames):
        b, n, c, h, w = transition.shape
        a = frames[:, :-1].flatten(0, 1)
        z = frames[:, 1:].flatten(0, 1)
        a = F.normalize(self.embedding(a).float(), dim=1)
        z = F.normalize(self.embedding(z).float(), dim=1)
        kernel = 2 * self.radius + 1
        candidates = F.unfold(z, kernel_size=kernel, padding=self.radius)
        candidates = candidates.view(b * n, a.shape[1], kernel * kernel, h, w)
        corr = (a.unsqueeze(2) * candidates).sum(dim=1)
        x = torch.cat([transition.flatten(0, 1), corr.to(transition.dtype)], dim=1)
        return self.fuse(x).unflatten(0, (b, n))


class NCCFusion(nn.Module):
    """Residual L2b fusion, preserving pre-trained L1 features at init."""
    def __init__(self, channels=128, windows=(15, 31)):
        super().__init__()
        self.windows = tuple(windows)
        self.body = nn.Sequential(
            nn.Conv2d(channels + len(windows), channels, 3, padding=1, bias=False),
            _norm(channels), nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, transition, video, dec_override=None):
        b, n, c, h, w = transition.shape
        ncc, valid = spatial_ncc(video, self.windows, (h, w))
        # Invalid measurements have zero contribution. A non-learnable 10x
        # rescaling resolves useful sub-percent NCC differences numerically.
        dec = ((1 - ncc) * valid.float() * 10).to(transition.dtype)
        if dec_override is not None:
            if dec_override.shape != dec.shape:
                raise ValueError(f'NCC override shape {dec_override.shape} != {dec.shape}')
            # Preserve the original video's validity mask. Zero means NO NCC
            # evidence, not NCC=0 (which would incorrectly mean decorrelation).
            dec = dec_override.to(device=dec.device, dtype=dec.dtype) * valid.to(dec.dtype)
        fused = self.body(torch.cat([transition, dec], dim=2).flatten(0, 1))
        return transition + fused.unflatten(0, (b, n)), ncc, valid


class LocalStream(nn.Module):
    """Reusable dense stream + disposable motion diagnostic head.

    Inputs: video [B,T,1,H,W], T>=2, intensities [0,1].
    Outputs: dense [B,T-1,C,ceil(H/4),ceil(W/4)],
             frame_dense [B,T,C,h,w] or None for L0,
             prediction [B,T-1,D] in physical task units.
    Diagnostic target conventions: source point transformed into next probe frame
    via inv(T_(t+1)) @ T_t. ``rigid`` outputs rotvec xyz (radians) then
    translation xyz (mm); elev_mag predicts abs(translation z) (mm).
    """
    TASK_DIMS = {'lateral': 1, 'inplane_rot': 1, 'rot_x': 1, 'rot_y': 1,
                 'elev_mag': 1, 'elev_signed': 1, 'rigid': 6}

    def __init__(self, arch='l1', task='lateral', channels=128, correlation_radius=4,
                 frame_norm=False):
        super().__init__()
        if arch not in ('l0', 'l1', 'l2', 'l2b'):
            raise ValueError(f'Unknown arch: {arch}')
        if task not in self.TASK_DIMS:
            raise ValueError(f'Unknown task: {task}')
        self.arch, self.task = arch, task
        self.encoder = Pair2D(channels) if arch == 'l0' else Temporal3D(channels, frame_norm)
        self.correlation = LocalCorrelation(channels, correlation_radius) if arch == 'l2' else None
        self.ncc_fusion = NCCFusion(channels) if arch == 'l2b' else None
        # A small spatially aware head; *do not* pool away dense maps sent to D4RT.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(channels * 16, 256), nn.GELU(),
            nn.Linear(256, self.TASK_DIMS[task]),
        )

    def forward(self, video, ncc_dec_override=None):
        if ncc_dec_override is not None and self.ncc_fusion is None:
            raise ValueError('NCC ablation is only supported for arch=l2b')
        if video.ndim != 5 or video.shape[1] < 2 or video.shape[2] != 1:
            raise ValueError('Expected video [B,T>=2,1,H,W]')
        if self.arch == 'l0':
            dense = self.encoder(video)
            frame_dense = None
        else:
            dense, frame_dense = self.encoder(video)
        if self.correlation is not None:
            dense = self.correlation(dense, frame_dense)
        ncc = ncc_valid = None
        if self.ncc_fusion is not None:
            dense, ncc, ncc_valid = self.ncc_fusion(dense, video, dec_override=ncc_dec_override)
        b, n = dense.shape[:2]
        pred = self.head(dense.flatten(0, 1)).unflatten(0, (b, n))
        if self.task == 'elev_mag':
            pred = F.softplus(pred)
        return {'dense': dense, 'frame_dense': frame_dense, 'prediction': pred,
                'ncc': ncc, 'ncc_valid': ncc_valid}


def regression_loss(prediction, target, task):
    """Huber on dimensionless component errors; scales fixed before experiments."""
    scales = {'lateral': [0.8], 'inplane_rot': [0.04],
              'rot_x': [0.04], 'rot_y': [0.04],
              'elev_mag': [0.8], 'elev_signed': [0.8],
              'rigid': [0.04, 0.04, 0.04, 0.8, 0.8, 0.8]}
    scale = prediction.new_tensor(scales[task])
    return F.smooth_l1_loss(prediction / scale, target / scale, beta=0.5)
