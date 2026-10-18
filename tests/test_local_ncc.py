"""CPU-only tests for NCC, L2b fusion, calibration and geometric dense labels."""
import json

import numpy as np
import pytest
import torch

from ct2eco.local_ncc import spatial_ncc, dense_abs_z_from_poses
from ct2eco.local_ncc_baseline import fit_decreasing_curve, predict_curve, evaluate_dense
from ct2eco.local_stream import LocalStream
from ct2eco.local_data import SweepWindows


def test_identity_ncc_and_independent_speckle():
    gen = torch.Generator().manual_seed(91)
    a = torch.rand((3, 1, 64, 64), generator=gen)
    b = torch.rand((3, 1, 64, 64), generator=gen)
    same = torch.stack((a, a), dim=1)
    diff = torch.stack((a, b), dim=1)
    s, sv = spatial_ncc(same)
    d, dv = spatial_ncc(diff)
    assert s.shape == (3, 1, 2, 16, 16)
    assert sv.all() and dv.all()
    assert (s - 1).abs().max() < 1e-4
    assert d.abs().mean() < 0.25
    assert s.mean() > d.mean() + 0.7


def test_zero_and_homogeneous_region_invalid_not_nan():
    z = torch.zeros(1, 2, 1, 32, 40)
    maps, valid = spatial_ncc(z)
    assert not valid.any()
    assert torch.isfinite(maps).all()
    assert (maps == 0).all()
    with pytest.raises(ValueError, match='odd'):
        spatial_ncc(z, windows=(10,))


def test_geometry_dense_dz_exact_and_signfree():
    p = torch.eye(4).repeat(2, 2, 1, 1)
    p[0, 1, 2, 3] = 0.8
    p[1, 1, 2, 3] = -0.8
    z = dense_abs_z_from_poses(p, (32, 40), torch.full((2, 2), 0.4), (8, 10))
    assert z.shape == (2, 1, 1, 8, 10)
    assert torch.allclose(z, torch.full_like(z, 0.8), atol=1e-6)
    theta = 0.08
    p[:] = torch.eye(4)
    p[:, 1, 1, 1] = np.cos(theta)
    p[:, 1, 1, 2] = -np.sin(theta)
    p[:, 1, 2, 1] = np.sin(theta)
    p[:, 1, 2, 2] = np.cos(theta)
    z = dense_abs_z_from_poses(p, (32, 40), torch.full((2, 2), 0.4), (8, 10))
    expected = np.sin(theta) * torch.arange(8)[:, None] * 4 * 0.4
    assert torch.allclose(z[0, 0, 0], expected.expand(8, 10), atol=1e-4)


def test_l2b_same_initial_dense_as_l1_and_gradients():
    torch.manual_seed(101)
    x = torch.rand(2, 3, 1, 32, 40)
    l1 = LocalStream('l1', 'elev_mag', channels=32)
    l2b = LocalStream('l2b', 'elev_mag', channels=32)
    l2b.encoder.load_state_dict(l1.encoder.state_dict())
    l2b.head.load_state_dict(l1.head.state_dict())
    out_1, out_2 = l1(x), l2b(x)
    assert torch.allclose(out_1['dense'], out_2['dense'], atol=1e-5)
    assert torch.allclose(out_1['prediction'], out_2['prediction'], atol=1e-5)
    assert out_2['ncc'].shape == (2, 2, 2, 8, 10)
    assert out_2['ncc_valid'].dtype == torch.bool
    out_2['prediction'].mean().backward()
    assert l2b.ncc_fusion.body[-1].weight.grad.abs().sum() > 0


def test_calibration_monotonic_and_valid_mask():
    x = np.linspace(0.1, 0.99, 800)
    y = (1-x) * 5
    curve = fit_decreasing_curve(x, y, bins=40)
    assert np.all(np.diff(curve[1]) <= 1e-9)
    assert np.mean(np.abs(predict_curve(x, curve) - y)) < 0.02
    vals = np.ones((2, 3, 4), dtype=np.float32)
    predicted = vals * 1.1
    valid = np.ones_like(vals, dtype=bool)
    valid[:, 0, 0] = False
    report = evaluate_dense(predicted, vals, valid, np.zeros((3, 4)))
    assert report['n_valid_pixels'] == 22
    assert abs(report['ncc_mae_mm'] - 0.1) < 1e-6
    assert report['ratio_vs_spatial_prior'] < 0.11


def test_spacing_is_exposed_only_for_supervision(tmp_path):
    md = tmp_path / 'metadata'
    md.mkdir()
    (md / 'splits.json').write_text(json.dumps({'train': ['A'], 'val': ['B'], 'test': []}))
    for case in ('A', 'B'):
        d = tmp_path / 'sweeps' / (case + '_sweep00')
        d.mkdir(parents=True)
        (d / 'sweep.json').write_text(json.dumps({'case_id': case, 'motion': 'elevational_pure',
                                                 'spacing_mm': [0.4, 0.5]}))
        np.save(d / 'frames.npy', np.full((5, 32, 40), 128, np.uint8))
        np.save(d / 'poses.npy', np.tile(np.eye(4, dtype=np.float32), (5, 1, 1)))
    ds = SweepWindows(tmp_path, 'train', task='elev_mag', frames=3, length=1,
                      expected_motion='elevational_pure', return_spacing=True)
    item = ds[0]
    assert len(item) == 4
    assert item[3].tolist() == pytest.approx([0.4, 0.5])
    plain = SweepWindows(tmp_path, 'train', task='elev_mag', frames=3, length=1)
    assert len(plain[0]) == 3
