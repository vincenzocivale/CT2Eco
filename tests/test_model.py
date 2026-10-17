import torch

from ct2eco.model import USD4RT, geo_loss


def inputs(B=2, T=3, N=5, H=200, W=150):
    g = torch.Generator().manual_seed(0)
    frames = torch.rand(B, T, 1, H, W, generator=g)
    src, ref = torch.randint(T, (B, N), generator=g), torch.randint(T, (B, N), generator=g)
    uv = torch.rand(B, N, 2, generator=g) * torch.tensor([W - 1, H - 1])
    return frames, torch.full((B, 2), 0.4), src, ref, uv


def test_shapes_and_variable_frames_queries():
    model = USD4RT(depth=2, dec_depth=2).eval()
    for T, N in ((3, 5), (6, 1)):
        frames, spacing, src, ref, uv = inputs(T=T, N=N)
        assert model(frames, spacing, src, ref, uv).shape == (2, N, 3)


def test_queries_are_independent():
    torch.manual_seed(0)
    model = USD4RT(depth=2, dec_depth=2).eval()
    frames, spacing, src, ref, uv = inputs(N=6)
    with torch.no_grad():
        full = model(frames, spacing, src, ref, uv)
        alone = model(frames, spacing, src[:, :1], ref[:, :1], uv[:, :1])
    assert torch.allclose(full[:, :1], alone, atol=1e-3)  # outputs are in mm (x50): float32 noise ~1e-5


def test_loss_backpropagates_to_encoder():
    model = USD4RT(depth=2, dec_depth=2)
    frames, spacing, src, ref, uv = inputs()
    geo_loss(model(frames, spacing, src, ref, uv), torch.randn(2, 5, 3) * 20).backward()
    assert model.encoder.net[0].weight.grad.abs().sum() > 0


def test_sign_invariant_loss_ignores_a_per_video_z_flip():
    gt = torch.randn(2, 7, 3) * 20
    flipped = gt.clone()
    flipped[0, :, 2] *= -1  # video 0 mirrored in elevation, video 1 exact
    assert geo_loss(flipped, gt, sign_invariant=True) < 1e-6 < geo_loss(flipped, gt)
    mixed = gt.clone()
    mixed[0, :3, 2] *= -1  # a flip of only some queries is not a global ambiguity: still penalised
    assert geo_loss(mixed, gt, sign_invariant=True) > 1e-3


def test_early_fusion_shapes_and_single_frame():
    model = USD4RT(depth=2, dec_depth=2, early_fusion=True).eval()
    for T in (1, 4):
        frames, spacing, src, ref, uv = inputs(T=T)
        assert model(frames, spacing, src, ref, uv).shape == (2, 5, 3)


def test_local_patch_reads_the_source_pixels():
    model = USD4RT(depth=2, dec_depth=2, local_patch=3).eval()
    frames, spacing, src, ref, uv = inputs(T=4)
    uv = uv.round()  # integer pixel centres: bilinear sampling returns the exact pixel values
    patch = model._pixel_patch(frames, src, uv)
    b, n = 1, 2
    u, v, t = int(uv[b, n, 0]), int(uv[b, n, 1]), int(src[b, n])
    assert torch.allclose(patch[b, n, 4], frames[b, t, 0, v, u] * 2 - 1, atol=1e-5)  # centre of the 3x3 patch
    assert model(frames, spacing, src, ref, uv).shape == (2, 5, 3)


def test_residual_zero_head_is_probe_did_not_move():
    model = USD4RT(depth=2, dec_depth=2, residual=True).eval()
    torch.nn.init.zeros_(model.head[1].weight)
    torch.nn.init.zeros_(model.head[1].bias)
    frames, spacing, src, ref, uv = inputs()
    out = model(frames, spacing, src, ref, uv)
    assert torch.allclose(out[..., 0], (uv[..., 0] - 149 / 2) * 0.4, atol=1e-4) and out[..., 2].abs().max() == 0


def test_frame_tokens_shapes_and_independence():
    torch.manual_seed(0)
    model = USD4RT(depth=2, dec_depth=2, frame_tokens=True).eval()
    frames, spacing, src, ref, uv = inputs(T=4, N=6)
    with torch.no_grad():
        full = model(frames, spacing, src, ref, uv)
        alone = model(frames, spacing, src[:, :1], ref[:, :1], uv[:, :1])
    assert full.shape == (2, 6, 3) and torch.allclose(full[:, :1], alone, atol=1e-3)
