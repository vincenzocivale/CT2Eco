"""L3 residual geometry, teacher isolation, controls and checkpoint protocol."""

import json

import pytest
import torch

from ct2eco.local_l3_residual import LocalL3Residual, DzResidual, refine_positive
from ct2eco.local_stream import LocalStream
from scripts.train_local_dense import DenseXYZHead
from scripts.train_local_l3 import sampled_case_ids, evaluate, verify_data_provenance
from tests.test_local_ncc_alignment import _tiny_root
from ct2eco.local_data import SweepWindows
from torch.utils.data import DataLoader


def _model(variant):
    torch.manual_seed(42)
    return LocalL3Residual(LocalStream('l1', 'rigid'), DenseXYZHead(), variant,
                           windows=(5, 9))


def test_residual_exactly_recovers_frozen_l1_at_init():
    clip = torch.rand(2, 5, 1, 32, 40)
    spacing = torch.tensor([[.4, .3], [.7, .5]])
    for variant in ('r0', 'r1', 'r2'):
        model = _model(variant)
        out = model(clip, spacing)
        assert out['prediction'].shape == (2, 4, 3, 8, 10)
        assert torch.allclose(out['prediction'], out['baseline'], atol=2e-5)
        assert torch.all(out['prediction'][:, :, 2] >= 0)
        assert torch.all(out['residual'] == 0)
        assert not any(p.requires_grad for p in model.encoder.parameters())
        assert not any(p.requires_grad for p in model.probe.parameters())


def test_all_modes_same_parameter_count_and_controls():
    models = [_model(x) for x in ('r0', 'r1', 'r2')]
    assert len({sum(p.numel() for p in m.refiner.parameters()) for m in models}) == 1
    clip = torch.rand(1, 5, 1, 32, 40)
    spacing = torch.tensor([[.4, .3]])
    outs = [m(clip, spacing) for m in models]
    assert torch.equal(outs[0]['ncc_aux'], torch.zeros_like(outs[0]['ncc_aux']))
    assert outs[1]['ncc_aux'][:, :, 4:].abs().sum() == 0
    assert torch.equal(outs[1]['ncc_aux'][:, :, :4], outs[2]['ncc_aux'][:, :, :4])
    assert torch.equal(outs[0]['baseline'], outs[1]['baseline'])
    assert torch.equal(outs[1]['baseline'], outs[2]['baseline'])


def test_positive_logits_and_gradients_only_for_refiner():
    model = _model('r2')
    model.train()
    assert not model.encoder.training and not model.probe.training
    clip = torch.rand(1, 5, 1, 32, 40)
    out = model(clip, torch.tensor([[.4, .3]]))
    assert torch.all(refine_positive(out['baseline'][:, :, 2:3], torch.zeros_like(out['residual'])) >= 0)
    out['prediction'][:, :, 2].mean().backward()
    assert any(p.grad is not None for p in model.refiner.parameters())
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.probe.parameters())


def test_pose_not_accepted_and_invalid_inputs_rejected():
    model = _model('r2')
    with pytest.raises(ValueError, match='five-frame'):
        model(torch.rand(1, 2, 1, 32, 40), torch.ones(1, 2))
    with pytest.raises(ValueError, match='spacing'):
        model(torch.rand(1, 5, 1, 32, 40), torch.ones(1, 2) * -1)
    with pytest.raises(TypeError):
        model(torch.rand(1, 5, 1, 32, 40), torch.ones(1, 2), torch.eye(4))
    with pytest.raises(ValueError, match='variant'):
        _model('incorrect')


def test_case_tracking_and_matched_baseline(tmp_path):
    root = _tiny_root(tmp_path)
    ds = SweepWindows(root, 'val', task='rigid', frames=5, length=4,
                      seed=900001, expected_motion='random', return_spacing=True)
    case_ids = sampled_case_ids(ds)
    assert case_ids == ['ct_b'] * 4
    result = evaluate(_model('r0'), DataLoader(ds, batch_size=2), 3., 'cpu', case_ids)
    assert result['n_ct'] == 1
    assert result['primary_pixels'] > 0
    assert abs(result['base_dz_mae_mm'] - result['refined_dz_mae_mm']) < 2e-5
    assert abs(result['delta_dz_vs_l1_mm']) < 2e-5
    assert result['base_component_mae_mm'][:2] == result['refined_component_mae_mm'][:2]
    assert result['by_ct']['ct_b']['pixels'] > 0


def test_case_leakage_and_teacher_dataset_guard(tmp_path):
    root = _tiny_root(tmp_path)
    checkpoint = tmp_path / 'base.pt'
    torch.save({'config': {'data': str(root)}}, checkpoint)
    verify_data_provenance(root, checkpoint)
    torch.save({'config': {'data': '/different/dataset'}}, checkpoint)
    with pytest.raises(ValueError, match='trained on'):
        verify_data_provenance(root, checkpoint)
    torch.save({'config': {}}, checkpoint)
    splits = root / 'metadata' / 'splits.json'
    bad = json.loads(splits.read_text())
    bad['val'] = bad['train']
    splits.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='split leakage'):
        verify_data_provenance(root, checkpoint)
