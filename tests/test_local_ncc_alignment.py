"""Geometry, no-pose estimated matching, fair support and train/val smoke."""

import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from ct2eco.local_ncc_alignment import (oracle_flow, estimated_flow,
                                        warp_target, aligned_ncc, three_way_ncc)
from scripts.eval_local_ncc_alignment import (collect, fit_train_calibration,
                                               report_validation, _sweep_paths)


def _poses(n=1, tx=0, ty=0):
    p = torch.eye(4).repeat(n, 2, 1, 1)
    p[:, 1, 0, 3] = tx
    p[:, 1, 1, 3] = ty
    return p


def _texture(h=64, w=68, seed=7):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(1, 1, h, w, generator=g)
    return .2 + .6 * F.avg_pool2d(a, 3, 1, 1)


def test_oracle_sign_spacing_identity_and_warp():
    p = _poses(tx=-1., ty=2.)
    flow, z = oracle_flow(p, torch.tensor([[.5, .25]]), (40, 44))
    assert torch.allclose(flow[:, 0], torch.full_like(flow[:, 0], 4.))
    assert torch.allclose(flow[:, 1], torch.full_like(flow[:, 1], -4.))
    assert not z.any()
    s = _texture(64, 68)
    t = F.pad(s[..., :, :-4], (4, 0))  # image shifted right by 4 px
    flow = torch.zeros(1, 2, 64, 68)
    flow[:, 0] = 4
    out, mask = warp_target(t, flow)
    assert (out - s).abs()[..., 6:-6, 6:-6].mean() < 1e-6
    assert mask[..., :, -4:].sum() == 0


def test_oracle_compensated_ncc_exceeds_fixed_for_image_shift():
    s = _texture()
    t = F.pad(s[..., :, :-4], (4, 0))
    vid = torch.stack((s, t), 1)
    p = _poses(tx=-1.)  # dx=0.25mm, relative +4 pixels
    spacing = torch.tensor([[.5, .25]])
    out = three_way_ncc(vid, p, spacing, windows=(7, 11),
                        downsample=2, radius=3, patch=5)
    mask = out['valid']
    assert mask.any()
    a = out['ncc']['oracle'].mean(1)[mask].mean()
    b = out['ncc']['fixed'].mean(1)[mask].mean()
    assert a > b + .2
    assert out['dz'].max() < 1e-6


def test_image_estimation_has_no_pose_input_and_recovers_translation():
    s = _texture(h=64, w=68)
    t = F.pad(s[..., :, :-4], (4, 0))
    flow, confidence = estimated_flow(s, t, downsample=2, radius=3, patch=7,
                                      minimum_score=.10)
    err = (flow[:, 0, 15:-15, 15:-15] - 4).abs().mean()
    assert err < 1.5, f'Poor estimated displacement: {err}'
    assert confidence[:, :, 15:-15, 15:-15].mean() > .5


def test_true_z_linear_spatial_map_not_pose_input_for_estimated():
    p = _poses()
    theta = .015
    p[0, 1, 2, 1] = theta
    p[0, 1, 1, 2] = -theta
    _, z = oracle_flow(p, torch.tensor([[.4, .3]]), (30, 34))
    # z changes with depth for an x-axis rotation
    assert z[0, 0, 25, 10] > z[0, 0, 5, 10]


def _tiny_root(tmp_path):
    root = tmp_path / 'synthetic'
    (root / 'metadata').mkdir(parents=True)
    (root / 'metadata' / 'splits.json').write_text(json.dumps({
        'train': ['ct_a'], 'val': ['ct_b'], 'test': ['ct_untouched']}))
    rng = np.random.default_rng(17)
    for case in ('ct_a', 'ct_b'):
        d = root / 'sweeps' / f'{case}_s0'
        d.mkdir(parents=True)
        frames = (rng.random((6, 32, 36)) * 255).astype(np.uint8)
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], 6, axis=0)
        poses[:, 0, 3] = np.linspace(0, -.3, 6)
        poses[:, 2, 3] = np.linspace(0, .4, 6)
        np.save(d / 'frames.npy', frames)
        np.save(d / 'poses.npy', poses)
        (d / 'sweep.json').write_text(json.dumps({'case_id': case,
                                                'motion': 'random',
                                                'spacing_mm': [0.5, 0.5]}))
    return root


def test_collect_and_score_never_open_test(tmp_path):
    root = _tiny_root(tmp_path)
    kwargs = dict(gap=1, pairs_per_sweep=2, batch_size=2,
                  device='cpu', windows=(5, 9), downsample=2, radius=1, patch=3)
    tr = collect(root, 'train', **kwargs)
    va = collect(root, 'val', **kwargs)
    assert set(tr['case_ids']) == {'ct_a'}
    assert set(va['case_ids']) == {'ct_b'}
    prior, curves, fit = fit_train_calibration(tr, max_dz=3,
                                              windows=(5, 9), max_pixels=1000)
    report = report_validation(va, prior, curves, max_dz=3, bootstraps=5)
    assert report['n_cases'] == 1
    assert report['primary_mae_mm']['oracle'] >= 0
    assert all(x is not None for x in report['primary_mae_mm'].values())
    assert fit['training_matched_pixels'] > 0


def test_case_leakage_rejected(tmp_path):
    root = _tiny_root(tmp_path)
    (root / 'metadata' / 'splits.json').write_text(json.dumps({
        'train': ['ct_a'], 'val': ['ct_a'], 'test': ['ct_untouched']}))
    with pytest.raises(ValueError, match='leakage'):
        _sweep_paths(root, 'val')
