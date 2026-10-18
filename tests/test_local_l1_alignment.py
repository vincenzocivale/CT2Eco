"""Frozen-L1 registration: unit/geometry, checkpoint guards, shared validation."""

import json

import numpy as np
import pytest
import torch
from torch import nn

from ct2eco.local_l1_alignment import (l1_inplane_flow, upsample_probe_mm,
                                        load_l1_dense_checkpoint)
from ct2eco.local_stream import LocalStream
from scripts.train_local_dense import DenseXYZHead
from scripts.eval_local_l1_alignment import collect, fit_calibration, evaluate
from tests.test_local_ncc_alignment import _tiny_root


def test_upsample_respects_convolution_cell_anchor():
    h, w = 28, 32
    field = torch.zeros(1, 2, 7, 8)
    field[0, 0] = 4 * torch.arange(8)[None, :]
    field[0, 1] = 4 * torch.arange(7)[:, None]
    raw = upsample_probe_mm(field, (h, w))
    assert torch.allclose(raw[0, 0, :25, :29], torch.arange(29).expand(25, 29).float(), atol=1e-5)
    assert torch.allclose(raw[0, 1, :25, :29], torch.arange(25)[:, None].expand(25, 29).float(), atol=1e-5)


class _DummyEncoder(nn.Module):
    def forward(self, clip):
        b = clip.shape[0]
        return {'dense': torch.ones(b, 4, 128, 8, 9, device=clip.device)}


class _DummyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv2d(128, 3, 1)
        nn.init.zeros_(self.net.weight)
        with torch.no_grad():
            self.net.bias[:] = torch.tensor([1., -2., 0.])


def test_l1_flow_mm_to_pixels_axis_order_and_sign():
    clip = torch.rand(2, 5, 1, 32, 36)
    spacings = torch.tensor([[.5, .25], [2., .5]])
    flow = l1_inplane_flow(_DummyEncoder(), _DummyHead(), clip, spacings)
    assert flow.shape == (2, 2, 32, 36)
    assert torch.allclose(flow[:, :, 4, 8], torch.tensor([[4., -4.], [2., -1.]]), atol=1e-6)


def test_reject_wrong_clip_and_spacing():
    with pytest.raises(ValueError, match='five-frame'):
        l1_inplane_flow(_DummyEncoder(), _DummyHead(), torch.rand(1, 2, 1, 32, 36), torch.ones(1, 2))
    with pytest.raises(ValueError, match='spacing'):
        l1_inplane_flow(_DummyEncoder(), _DummyHead(), torch.rand(1, 5, 1, 32, 36), -torch.ones(1, 2))


def _save_checkpoint(path, head='xyz'):
    enc = LocalStream('l1', 'rigid')
    dense = DenseXYZHead()
    torch.save({'encoder': enc.state_dict(), 'dense_probe': dense.state_dict(),
                'config': {'head': head, 'arch': 'l1', 'motion': 'random', 'frames': 5}}, path)


def test_checkpoint_check_and_read(tmp_path):
    ckpt = tmp_path / 'ckpt.pt'
    _save_checkpoint(ckpt)
    enc, head = load_l1_dense_checkpoint(ckpt)
    assert not enc.training and not head.training
    assert not any(p.requires_grad for p in enc.parameters())
    assert not any(p.requires_grad for p in head.parameters())
    _save_checkpoint(ckpt, 'dz')
    with pytest.raises(ValueError, match='Checkpoint must be'):
        load_l1_dense_checkpoint(ckpt)


def test_four_way_matched_support_and_val_only(tmp_path):
    root = _tiny_root(tmp_path)
    ckpt = tmp_path / 'frozen.pt'
    _save_checkpoint(ckpt)
    enc, probe = load_l1_dense_checkpoint(ckpt)
    options = dict(batch_size=2, device='cpu', windows=(5, 9), encoder=enc, probe=probe,
                   downsample=2, radius=1, patch=3)
    train = collect(root, 'train', pairs_per_sweep=2, **options)
    val = collect(root, 'val', pairs_per_sweep=2, **options)
    assert set(train['case_ids']) == {'ct_a'}
    assert set(val['case_ids']) == {'ct_b'}
    assert all(val[k].shape == val['fixed'].shape for k in ('oracle','estimated','l1'))
    assert val['valid'].any()
    prior, curves, info = fit_calibration(train, windows=(5, 9), max_dz_mm=3, max_pixels=1000)
    report = evaluate(val, prior, curves, max_dz_mm=3, bootstraps=5)
    assert len(report['primary_mae_mm']) == 5
    assert report['n_ct'] == 1
    assert report['n_pairs'] == 2
    assert report['flow_epe_px']['l1'] >= 0
    assert info['train_calibration_pixels'] > 0
    assert set(report['by_ct']) == {'ct_b'}
