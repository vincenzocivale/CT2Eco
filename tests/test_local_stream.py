"""Tests for isolated local stream experiments. CPU-only, no CT downloads."""

import torch
import pytest

from ct2eco.local_stream import LocalStream, regression_loss
from ct2eco.local_data import phantom_batch, _labels_from_poses


@pytest.mark.parametrize('arch', ['l0', 'l1', 'l2'])
def test_dense_shape_no_temporal_downsampling(arch):
    torch.manual_seed(0)
    x = torch.rand(2, 4, 1, 32, 40)
    net = LocalStream(arch, 'lateral', channels=32, correlation_radius=2)
    z = net(x)
    assert z['dense'].shape == (2, 3, 32, 8, 10)
    assert z['prediction'].shape == (2, 3, 1)
    assert (z['frame_dense'] is None) == (arch == 'l0')
    if arch != 'l0':
        assert z['frame_dense'].shape == (2, 4, 32, 8, 10)
    z['prediction'].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.encoder.parameters())


@pytest.mark.parametrize('task', ['lateral', 'inplane_rot', 'elev_mag', 'rigid'])
def test_phantom_targets_and_frames(task):
    g = torch.Generator().manual_seed(0)
    x, y, p = phantom_batch(2, frames=4, image_size=(24, 32), task=task, generator=g)
    assert x.shape == (2, 4, 1, 24, 32)
    assert y.shape == (2, 3, LocalStream.TASK_DIMS[task])
    assert y.abs().max() < 10
    assert (x >= 0).all() and (x <= 1).all()
    assert torch.allclose(y, _labels_from_poses(p, task))
    if task == 'elev_mag':
        assert (y >= 0).all()
    if task == 'lateral':
        # Motion must exist in pixels, not merely in pose labels.
        assert (x[:, 1] - x[:, 0]).abs().mean() > 0.005


def test_probe_translations_reversal_and_loss():
    p = torch.eye(4).repeat(1, 3, 1, 1)
    p[:, 1, 0, 3] = 1.0
    p[:, 2, 0, 3] = 3.0
    y = _labels_from_poses(p, 'lateral')
    assert torch.allclose(y[0, :, 0], torch.tensor([-1.0, -2.0]))
    flipped = _labels_from_poses(p.flip(1), 'lateral').flip(1)
    assert torch.allclose(flipped, -y)
    assert regression_loss(y, y, 'lateral') == 0


def test_elevation_sign_ambiguity_and_absolute_target():
    p = torch.eye(4).repeat(2, 2, 1, 1)
    p[0, 1, 2, 3] = 0.6
    p[1, 1, 2, 3] = -0.6
    out = _labels_from_poses(p, 'elev_mag')
    assert torch.allclose(out[0], out[1], atol=1e-6)


def test_opposite_motion_same_anatomy_different_video():
    gen = torch.Generator().manual_seed(11)
    x, target, _ = phantom_batch(2, frames=4, image_size=(32, 32),
                                 task='lateral', generator=gen,
                                 signed_speed_px=[-1.5, 1.5], shared_volume=True)
    assert torch.allclose(x[0, 0], x[1, 0])
    assert not torch.allclose(x[0, 2], x[1, 2])
    assert torch.allclose(target[0], -target[1])


def test_ct2eco_sweep_loader_patient_split_and_sign(tmp_path):
    import json
    import numpy as np
    from ct2eco.local_data import SweepWindows

    md = tmp_path / 'metadata'
    md.mkdir()
    (md / 'splits.json').write_text(json.dumps({'train': ['A'], 'val': ['B'], 'test': []}))
    for case in ('A', 'B'):
        directory = tmp_path / 'sweeps' / (case + '_sweep00')
        directory.mkdir(parents=True)
        (directory / 'sweep.json').write_text(json.dumps({'case_id': case}))
        np.save(directory / 'frames.npy', np.zeros((5, 24, 32), np.uint8))
        poses = np.tile(np.eye(4, dtype=np.float32), (5, 1, 1))
        poses[:, 0, 3] = np.arange(5) * 0.4
        np.save(directory / 'poses.npy', poses)
    train = SweepWindows(tmp_path, 'train', task='lateral', frames=3, length=2)
    val = SweepWindows(tmp_path, 'val', task='lateral', frames=3, length=2)
    assert len(train.entries) == len(val.entries) == 1
    assert train.entries[0].name.startswith('A') and val.entries[0].name.startswith('B')
    clip, gt, _ = train[0]
    assert clip.shape == (3, 1, 24, 32)
    assert torch.allclose(gt[:, 0], torch.tensor([-0.4, -0.4]), atol=1e-6)

@pytest.mark.parametrize('task', ['elev_signed', 'rot_x', 'rot_y'])
def test_new_signed_and_rotation_target_conventions(task):
    p = torch.eye(4).repeat(1, 2, 1, 1)
    if task == 'elev_signed':
        p[0, 1, 2, 3] = 0.7
        expected = -0.7
    else:
        from scipy.spatial.transform import Rotation
        index = 0 if task == 'rot_x' else 1
        axis = [0., 0., 0.]
        axis[index] = 0.03
        p[0, 1, :3, :3] = torch.tensor(Rotation.from_rotvec(axis).as_matrix(), dtype=p.dtype)
        expected = -0.03
    lab = _labels_from_poses(p, task)
    assert lab.shape == (1, 1, 1)
    assert torch.allclose(lab[0, 0, 0], torch.tensor(expected), atol=1e-5)
    inv = _labels_from_poses(p.flip(1), task)
    assert torch.allclose(inv, -lab, atol=1e-5)
    from ct2eco.local_stream import regression_loss
    assert regression_loss(lab, lab, task).item() == 0


def test_geometric_loss_exact_inverse_gradients_and_baselines():
    from ct2eco.local_geometry import point_error_mm, point_loss, transform_probe_points
    torch.manual_seed(0)
    gt = torch.tensor([[0., 0., 0.03, 0.4, -0.7, 0.1],
                       [0.02, 0., 0., 0., 0.1, -0.7]])
    pred = gt.clone().detach().requires_grad_()
    assert point_error_mm(pred, gt).max().item() < 2e-5
    assert point_loss(pred, gt).item() < 2e-5
    pred2 = torch.zeros_like(gt, requires_grad=True)
    error = point_loss(pred2, gt)
    assert error.item() > 0.4
    error.backward()
    assert torch.isfinite(pred2.grad).all() and pred2.grad.abs().sum() > 1e-3
    # Exact rigid rotation, not first-order approximation.
    p = torch.tensor([[0., 0., 1.57079632679, 0., 0., 0.]])
    o = transform_probe_points(p, ((1., 0., 0.),))
    assert torch.allclose(o[0, 0], torch.tensor([0., 1., 0.]), atol=1e-5)


def test_evaluate_reports_ablations_per_component_and_geometric_priors():
    from scripts.train_local import evaluate

    class Stub(torch.nn.Module):
        def forward(self, x):
            # output is deterministic but depends on image pixels
            z = x.mean(dim=(2, 3, 4))[:, :-1]
            y = torch.zeros(x.shape[0], x.shape[1] - 1, 6, device=x.device)
            y[..., 3] = z
            return {'prediction': y}

    images = torch.stack([torch.ones(3, 1, 16, 16), torch.zeros(3, 1, 16, 16)])
    gt = torch.zeros(2, 2, 6)
    gt[..., 3] = 0.5
    a = evaluate(Stub(), [(images, gt)], 'rigid', torch.zeros(6))
    for key in ('point_mae_mm', 'point_blind_mae_mm', 'point_identity_mae_mm',
                'point_zeros_mae_mm', 'point_other_video_mae_mm',
                'point_time_shuffle_mae_mm'):
        assert key in a and a[key] >= 0
    for condition in ('normal', 'zeros', 'other_video', 'time_shuffle'):
        assert len(a[f'per_component_{condition}_mae']) == 6
    assert a['point_blind_mae_mm'] == pytest.approx(0.5, abs=1e-5)
    assert a['point_identity_mae_mm'] == pytest.approx(0.5, abs=1e-5)
    assert a['point_mae_mm'] > 0.0


def test_require_pure_ct_motion_metadata(tmp_path):
    import json
    import numpy as np
    from ct2eco.local_data import SweepWindows
    (tmp_path / 'metadata').mkdir()
    (tmp_path / 'metadata' / 'splits.json').write_text(json.dumps({'train': ['A'], 'val': ['B'], 'test': ['C']}))
    for case, motion in [('A', 'rot_x'), ('B', 'random'), ('C', 'rot_x')]:
        d = tmp_path / 'sweeps' / case
        d.mkdir(parents=True)
        (d / 'sweep.json').write_text(json.dumps({'case_id': case, 'motion': motion}))
    SweepWindows(tmp_path, 'train', task='rot_x', frames=3, expected_motion='rot_x')
    with pytest.raises(ValueError, match='pure motion'):
        SweepWindows(tmp_path, 'val', task='rot_x', frames=3, expected_motion='rot_x')

@pytest.mark.parametrize('motion,expected_axis', [
    ('elevational_pure', 'z'), ('rot_x', 'x'), ('rot_y', 'y'), ('rot_z', 'z')
])
def test_pure_ct_sweep_motion_modes_are_geometrically_pure(motion, expected_axis, monkeypatch):
    import numpy as np
    from ct2eco import sweep

    class FakeCase:
        def __init__(self):
            self.seg = np.ones((32, 32, 32), np.uint8)
            self.label_map = {'background': 0}
            self.affine = np.eye(4)

    # Replace skin lookup so the test focuses on SE(3) kinematics, not CT anatomy.
    monkeypatch.setattr(sweep, 'skin_point', lambda case, x, z, step=0.5: np.array([x, 30., z]))
    p, _ = sweep.make_trajectory(FakeCase(), np.random.default_rng(42), 6,
                                 targets=[], motion=motion)
    assert p is not None
    relative = np.linalg.inv(p[1:]) @ p[:-1]
    from scipy.spatial.transform import Rotation
    rv = Rotation.from_matrix(relative[:, :3, :3]).as_rotvec()
    tr = relative[:, :3, 3]
    if motion == 'elevational_pure':
        assert np.abs(rv).max() < 1e-6
        assert np.abs(tr[:, :2]).max() < 1e-5
        assert np.max(np.abs(tr[:, 2])) > 1e-3
    else:
        assert np.abs(tr).max() < 1e-5
        axis = {'x': 0, 'y': 1, 'z': 2}[expected_axis]
        assert np.abs(np.delete(rv, axis, axis=1)).max() < 1e-6
        assert np.max(np.abs(rv[:, axis])) > 1e-5


def test_expected_motion_for_strict_elevation_mode():
    from types import SimpleNamespace
    from scripts.train_local import _expected_motion
    assert _expected_motion(SimpleNamespace(require_pure_motion=True,
                                            task='elev_mag')) == 'elevational_pure'
    assert _expected_motion(SimpleNamespace(require_pure_motion=True,
                                            task='elev_signed')) == 'elevational_pure'
