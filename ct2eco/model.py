"""US-D4RT v0.1: frame encoder -> VGGT-style alternating attention -> D4RT-style independent point queries.

Input: video only (frames + probe calibration). Poses are never an input.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG, GRID = 224, 14  # 16 px patches, same token grid as a ViT-B/16
COORD_SCALE_MM = 50.0  # fixed metric normalisation of the regressed coordinates
SMOOTH_L1_BETA_MM = 1.0


def fourier(x, n_freq, min_period, max_period):
    """(..., D) -> (..., D * 2 * n_freq) sin/cos features with log-spaced periods in the units of x."""
    periods = torch.logspace(math.log10(min_period), math.log10(max_period), n_freq, device=x.device)
    a = x[..., None].float() * (2 * math.pi / periods)
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


def pixel_mm(uv, spacing, width):
    """Torch twin of geometry.pixel_to_probe on the plane: (..., 2) pixels -> (..., 2) (x, y) mm. spacing = (dy, dx)."""
    return torch.stack([(uv[..., 0] - (width - 1) / 2) * spacing[..., 1], uv[..., 1] * spacing[..., 0]], -1)


class ConvPatchEncoder(nn.Module):
    """From-scratch CNN, stride 16: (N, in_ch, 224, 224) -> (N, 196, dim)."""

    def __init__(self, dim, in_ch=1):
        super().__init__()
        chs, layers = [in_ch, 48, 96, 192, dim], []
        for cin, cout in zip(chs[:-1], chs[1:]):
            layers += [nn.Conv2d(cin, cout, 3, 2, 1), nn.GroupNorm(8, cout), nn.GELU(),
                       nn.Conv2d(cout, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.GELU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).flatten(2).transpose(1, 2)


class Block(nn.Module):
    """Pre-norm attention + MLP. With ctx it is cross-attention (queries never attend to each other)."""

    def __init__(self, dim, heads, cross=False):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim) if cross else None
        self.q, self.kv, self.proj = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, ctx=None):
        h = self.norm1(x)
        c = h if ctx is None else self.norm_ctx(ctx)
        split = lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        k, v = self.kv(c).chunk(2, -1)
        a = F.scaled_dot_product_attention(split(self.q(h)), split(k), split(v))
        x = x + self.proj(a.transpose(1, 2).flatten(-2))
        return x + self.mlp(self.norm2(x))


class USD4RT(nn.Module):
    def __init__(self, dim=384, heads=6, depth=4, dec_depth=4, n_freq=16, early_fusion=False, local_patch=0,
                 residual=False, frame_tokens=False):
        """early_fusion: each frame enters the encoder stacked with its neighbours [t-1, t, t+1], so speckle
        decorrelation and in-plane motion are compared pixel by pixel instead of only between 16 px tokens.
        local_patch: odd k > 0 adds the raw k x k source-frame pixels around (u, v) at native resolution to each
        query, as D4RT does (its Tab. 7: removing the patch doubles the pose error).
        residual: the head predicts the displacement from the source point (x_s, y_s, 0), so a zero output is
        'probe did not move' and all capacity goes to the motion.
        frame_tokens: one learned register token per frame joins the trunk (like VGGT's camera token, but with no
        loss of its own); each query also gets g[t_r] - g[t_s]. The trunk can then store a per-frame cumulative
        state, so 'motion x frame gap' becomes a difference instead of a product the decoder must compute."""
        super().__init__()
        self.f = n_freq
        self.early_fusion = early_fusion
        self.encoder = ConvPatchEncoder(dim, 3 if early_fusion else 1)
        self.patch_pos = nn.Linear(2 * 2 * n_freq, dim)
        self.frame_pos = nn.Linear(2 * n_freq, dim)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))  # even: frame-wise, odd: global
        self.query_in = nn.Linear(4 * 2 * n_freq + dim, dim)
        self.local_patch = local_patch
        self.residual = residual
        self.frame_token = nn.Parameter(torch.randn(1, 1, 1, dim) * 0.02) if frame_tokens else None
        self.delta_in = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim)) if frame_tokens else None
        self.patch_in = nn.Linear(local_patch**2, dim) if local_patch else None
        self.decoder = nn.ModuleList(Block(dim, heads, cross=True) for _ in range(dec_depth))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))

    def _mm_feat(self, uv, spacing, width):
        return fourier(pixel_mm(uv, spacing, width), self.f, 1.0, 512.0)

    def _frame_feat(self, t):
        return fourier(t[..., None], self.f, 2.0, 256.0)

    def encode(self, frames, spacing):
        """frames (B, T, 1, H, W) in [0, 1], spacing (B, 2) mm -> shared tokens (B, T, 196, D)."""
        B, T, _, H, W = frames.shape
        x = F.interpolate(frames.flatten(0, 1), (IMG, IMG), mode="bilinear", align_corners=False) * 2 - 1
        if self.early_fusion:
            t = torch.arange(T, device=x.device)
            x = x.unflatten(0, (B, T))[:, :, 0]
            x = torch.stack([x[:, (t - 1).clamp(min=0)], x, x[:, (t + 1).clamp(max=T - 1)]], 2).flatten(0, 1)
        tok = self.encoder(x).unflatten(0, (B, T))
        # patch centres back in original pixels, then mm: the metric PE survives the anisotropic resize
        c = (torch.arange(GRID, device=x.device) + 0.5) / GRID
        gy, gx = torch.meshgrid(c * H - 0.5, c * W - 0.5, indexing="ij")
        uv = torch.stack([gx, gy], -1).flatten(0, 1)  # (196, 2)
        tok = tok + self.patch_pos(self._mm_feat(uv[None], spacing[:, None], W))[:, None]
        if self.frame_token is not None:
            tok = torch.cat([self.frame_token.expand(B, T, 1, -1), tok], 2)  # index 0 of every frame
        tok = tok + self.frame_pos(self._frame_feat(torch.arange(T, device=x.device)))[None, :, None]
        L = tok.shape[2]
        for i, blk in enumerate(self.blocks):
            if i % 2 == 0:
                tok = blk(tok.flatten(0, 1)).unflatten(0, (B, T))
            else:
                tok = blk(tok.flatten(1, 2)).unflatten(1, (T, L))
        return tok  # (B, T, [1 +] 196, D)

    def _pixel_patch(self, frames, src, uv):
        """(B, N, k*k) raw source-frame pixels around uv, bilinear, at the original resolution."""
        B, T, _, H, W = frames.shape
        r = torch.arange(self.local_patch, device=uv.device) - self.local_patch // 2
        dy, dx = torch.meshgrid(r, r, indexing="ij")
        pts = uv[:, :, None] + torch.stack([dx, dy], -1).flatten(0, 1)  # (B, N, k*k, 2)
        t = src[:, :, None].expand(-1, -1, pts.shape[2])
        grid = torch.stack([(pts[..., 0] + 0.5) / W, (pts[..., 1] + 0.5) / H, (t + 0.5) / T], -1) * 2 - 1
        out = F.grid_sample(frames.transpose(1, 2), grid[:, None].to(frames.dtype), align_corners=False, padding_mode="zeros")
        return out[:, 0, 0] * 2 - 1  # (B, N, k*k), same [-1, 1] range as the encoder input

    def decode(self, tok, spacing, image_hw, src, ref, uv, frames=None):
        """tok (B, T, 196, D); src, ref (B, N) frame indices; uv (B, N, 2) source pixels -> (B, N, 3) mm in the ref probe frame.
        frames is needed only with local_patch."""
        B, T = tok.shape[:2]
        H, W = image_hw
        fmap = tok[:, :, -GRID * GRID:].unflatten(2, (GRID, GRID)).permute(0, 4, 1, 2, 3)  # (B, D, T, 14, 14)
        grid = torch.stack([(uv[..., 0] + 0.5) / W, (uv[..., 1] + 0.5) / H, (src + 0.5) / T], -1) * 2 - 1
        local = F.grid_sample(fmap, grid[:, None, None].to(fmap.dtype), align_corners=False, padding_mode="border")
        local = local[:, :, 0, 0].transpose(1, 2)  # (B, N, D), bilinear in space, exact in time
        q = self.query_in(torch.cat([self._mm_feat(uv, spacing[:, None], W),
                                     self._frame_feat(src), self._frame_feat(ref), local.float()], -1).to(tok.dtype))
        if self.local_patch:
            q = q + self.patch_in(self._pixel_patch(frames, src, uv).to(q.dtype))
        if self.frame_token is not None:
            g, b = tok[:, :, 0], torch.arange(B, device=tok.device)[:, None]
            q = q + self.delta_in(g[b, ref] - g[b, src])
        mem = tok.flatten(1, 2)
        for blk in self.decoder:
            q = blk(q, mem)
        out = self.head(q).float() * COORD_SCALE_MM
        if self.residual:
            xy = pixel_mm(uv, spacing[:, None], W)
            out = out + torch.cat([xy, torch.zeros_like(xy[..., :1])], -1)
        return out

    def forward(self, frames, spacing, src, ref, uv):
        tok = self.encode(frames, spacing)
        return self.decode(tok, spacing, frames.shape[-2:], src, ref, uv, frames)


def geo_loss(pred_mm, gt_mm, sign_invariant=False, beta_mm=SMOOTH_L1_BETA_MM):
    """SmoothL1 on (B, N, 3) coordinates / COORD_SCALE_MM.

    sign_invariant: per video, the better of pred and pred with z negated. Diagnostic for the elevational
    direction ambiguity (speckle tells how far the plane moved, not which way): it is one unknown per video.
    """
    loss = lambda p: F.smooth_l1_loss(p / COORD_SCALE_MM, gt_mm / COORD_SCALE_MM, reduction="none",
                                      beta=beta_mm / COORD_SCALE_MM).mean((1, 2))
    per_video = loss(pred_mm)
    if sign_invariant:
        per_video = torch.minimum(per_video, loss(pred_mm * pred_mm.new_tensor([1, 1, -1])))
    return per_video.mean()
