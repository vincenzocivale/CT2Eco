"""T1 query geometry, frozen-feature controls, loss and aggregation."""
import pytest
import torch

from ct2eco.local_query_t1 import (PointQueryReadout, aggregate_query_errors,
                                   finalize_query_errors, loss_point_queries, sample_map)
from ct2eco.local_stream import LocalStream
from ct2eco.local_transfer import project_queries
from scripts.train_local_query_t1 import decode, make_frozen_encoder


def test_query_head_identity_init_and_shapes():
    torch.manual_seed(17)
    decoder = PointQueryReadout()
    q = torch.tensor([[3.,4.],[6.,7.]])
    for gap in (1,2,4):
        pred = decoder(q, gap, (96,96), 2)
        assert pred.shape == (2,2,3)
        assert torch.allclose(pred[..., :2],q[None]*4)
        assert bool((pred[...,2]>=0).all())
    with pytest.raises(ValueError):
        decoder(q,3,(96,96),2)
    with pytest.raises(ValueError):
        decoder(torch.tensor([[999.,1.]]),1,(96,96),2)


def test_sample_map_exact_cells_and_subpixel():
    f = torch.arange(25.).reshape(1,1,5,5)
    q = torch.tensor([[2.,3.],[1.5,1.]])
    result = sample_map(f,q)
    assert result.shape == (1,2,1)
    assert torch.allclose(result[0,:,0],torch.tensor([17., 6.5]),atol=1e-5)


def test_q0_q1_q2_initialization_equivalent_and_frozen():
    enc = LocalStream('l1','rigid').eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    random = make_frozen_encoder('q1',enc,'cpu',random_seed=314159)
    assert all(not p.requires_grad for p in random.parameters())
    torch.manual_seed(42)
    models = [PointQueryReadout() for _ in range(3)]
    initial = models[0].state_dict()
    for model in models[1:]:
        model.load_state_dict(initial)
    x = torch.rand(1,5,1,32,32)
    q = torch.tensor([[3,3],[4,4]])
    predictions = [decode(models[0],None,x,q,1),
                   decode(models[1],random,x,q,1),
                   decode(models[2],enc,x,q,1)]
    assert all(torch.allclose(predictions[0],y,atol=1e-5) for y in predictions[1:])
    loss = predictions[2][...,0].sum() + predictions[2][...,2].sum()
    loss.backward()
    assert any(p.grad is not None for p in models[2].parameters())
    assert all(p.grad is None for p in enc.parameters())


def test_labels_excluded_from_model_inputs_and_loss_masked():
    model = PointQueryReadout()
    poses = torch.eye(4).repeat(1,5,1,1)
    poses[:,1,0,3] = 1.
    spacing = torch.tensor([[.5,.25]])
    q = torch.tensor([[3,3],[5,5]])
    uv,z,ok = project_queries(poses,spacing,q,(96,96),0,1)
    pred = model(q,1,(96,96),1)
    loss = loss_point_queries(pred,uv,z,ok)
    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert torch.allclose(uv[0],q*4 + torch.tensor([-4.,0.]))
    with pytest.raises(ValueError):
        loss_point_queries(pred,uv,z,torch.zeros_like(ok))


def test_aggregation_band_counts_and_metrics():
    target = torch.tensor([[[10.,10.],[15.,15.],[20.,20.]]])
    dz = torch.tensor([[.1,1.,5.]])
    valid = torch.tensor([[True,True,False]])
    pred = torch.cat([target+torch.tensor([2.,0.]), dz[...,None]+.1],-1)
    a={}
    aggregate_query_errors(a,pred,target,dz,valid,overlap_mm=.5)
    result=finalize_query_errors(a)
    assert result['fov_all']['n_points']==2
    assert result['proxy_overlap']['n_points']==1
    assert result['z_0.5_1.5']['n_points']==1
    assert abs(result['fov_all']['epe_px']-2)<1e-6
    assert abs(result['fov_all']['abs_dz_mae_mm']-.1)<1e-5
