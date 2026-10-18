"""New incremental follow-up: NCC-only ablations + mixed dense geometry."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ct2eco.local_ncc import dense_motion_from_poses, dense_abs_z_from_poses
from ct2eco.local_stream import LocalStream
from scripts.train_local_dense import DenseXYZHead, evaluate, loss_dense, supervision, train_spatial_prior


def test_dense_motion_identity_and_exact_translation():
    ps = torch.eye(4).repeat(2, 3, 1, 1)
    spacing = torch.full((2, 2), 0.4)
    gt, mask = dense_motion_from_poses(ps, (32, 40), spacing, (8, 10), return_fov=True)
    assert gt.shape == (2, 2, 3, 8, 10)
    assert torch.allclose(gt, torch.zeros_like(gt))
    assert mask.all()
    ps[:, 1, :3, 3] = torch.tensor([0.8, -0.4, 1.2])
    ps[:, 2, :3, 3] = torch.tensor([1.6, -0.8, 2.4])
    gt, mask = dense_motion_from_poses(ps, (32, 40), spacing, (8, 10), return_fov=True)
    for i, desired in enumerate((-0.8, 0.4, 1.2)):
        assert torch.allclose(gt[:, :, i], torch.full_like(gt[:, :, i], desired), atol=1e-5)
    assert mask.float().mean() < 1.
    assert mask.float().mean() > 0.6
    assert torch.allclose(gt[:, :, 2:3], dense_abs_z_from_poses(ps, (32, 40), spacing, (8, 10)))


def test_dense_motion_rotation_matches_exact_coordinate_map():
    theta = 0.04
    p = torch.eye(4).repeat(1, 2, 1, 1)
    p[0, 1, 1, 1] = np.cos(theta)
    p[0, 1, 1, 2] = -np.sin(theta)
    p[0, 1, 2, 1] = np.sin(theta)
    p[0, 1, 2, 2] = np.cos(theta)
    spacing = torch.tensor([[0.4, 0.4]])
    gt = dense_motion_from_poses(p, (32, 40), spacing, (8, 10))
    y = torch.arange(8) * 4 * 0.4
    assert torch.allclose(gt[0, 0, 0], torch.zeros_like(gt[0, 0, 0]), atol=1e-5)
    assert torch.allclose(gt[0, 0, 1, :, 0], (np.cos(theta)-1) * y, atol=1e-5)
    assert torch.allclose(gt[0, 0, 2, :, 0], np.sin(theta) * y, atol=1e-5)


def test_ncc_only_override_keeps_l1_encoder_fixed():
    torch.manual_seed(33)
    model = LocalStream('l2b', 'rigid', channels=32)
    x = torch.rand(2, 3, 1, 32, 40)
    model.eval()
    with torch.no_grad():
        base = model(x)
        # Newly initialized NCC head is zero-initialized; only trainable fusion
        # makes an override affect the output.
        model.ncc_fusion.body[-1].weight.fill_(0.02)
        baseline = model(x)
        dec_zero = torch.zeros_like(baseline['ncc'])
        override = model(x, ncc_dec_override=dec_zero)
    assert torch.allclose(base['frame_dense'], baseline['frame_dense'])
    assert torch.allclose(base['frame_dense'], override['frame_dense'])
    assert torch.allclose(baseline['ncc_valid'], override['ncc_valid'])
    assert not torch.allclose(baseline['dense'], override['dense'])
    with pytest.raises(ValueError, match='shape'):
        model(x, ncc_dec_override=torch.zeros(1, 1))
    with pytest.raises(ValueError, match='arch=l2b'):
        LocalStream('l1', 'rigid', channels=32)(x, ncc_dec_override=dec_zero)


def test_xyz_head_gradient_and_supervision():
    head = DenseXYZHead(channels=32)
    data = torch.randn(2, 3, 32, 8, 10, requires_grad=True)
    pred = head(data)
    assert pred.shape == (2, 3, 3, 8, 10)
    assert (pred[:, :, 2] >= 0).all()
    gt = torch.zeros_like(pred)
    mask = torch.ones(2, 3, 1, 8, 10, dtype=torch.bool)
    loss_dense(pred, gt, mask, 'xyz').backward()
    assert data.grad.abs().sum() > 0
    x = torch.rand(2, 3, 1, 32, 40)
    ps = torch.eye(4).repeat(2, 3, 1, 1)
    sp = torch.full((2, 2), 0.4)
    gt, valid = supervision(x, ps, sp, (8, 10), head_type='xyz')
    assert gt.shape == (2, 2, 3, 8, 10)
    assert valid.shape == (2, 2, 1, 8, 10)


def test_train_only_prior_and_eval_xyz_smoke():
    torch.manual_seed(19)
    x = torch.rand(2, 3, 1, 32, 40)
    ps = torch.eye(4).repeat(2, 3, 1, 1)
    ps[:, 1, 2, 3] = 0.7
    ps[:, 2, 2, 3] = 1.4
    sp = torch.full((2, 2), 0.4)
    batches = [(x, torch.empty(2), ps, sp)]
    prior = train_spatial_prior(batches, torch.device('cpu'), 'xyz', 3.)
    assert prior.shape == (1, 1, 3, 8, 10)
    net = LocalStream('l2b', 'rigid', channels=32)
    head = DenseXYZHead(channels=32)
    metrics = evaluate(net, head, batches, prior, 3., 'cpu', head_type='xyz', ncc_ablation=True)
    assert len(metrics['component_normal_mae_mm']) == 3
    assert metrics['evaluated_pixels'] > 0
    for name in ('ncc_zero_mae_mm', 'ncc_other_mae_mm', 'ncc_spatial_roll_mae_mm'):
        assert np.isfinite(metrics[name])
