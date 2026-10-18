import torch
import pytest

from ct2eco.local_transfer import (query_positions, project_queries, match_descriptors,
                                   raw_patch_descriptors, compose_l1_displacements,
                                   query_metrics)


def test_query_geometry_identity_and_translation():
    pp = torch.eye(4).repeat(1, 5, 1, 1)
    sp = torch.tensor([[0.5, 0.25]])
    q = torch.tensor([[4, 6], [10, 8]])
    xy, z, fov = project_queries(pp, sp, q, (100, 100), 0, 1)
    assert torch.allclose(xy[0], q.float() * 4, atol=1e-5)
    assert torch.all(z == 0) and bool(fov.all())
    # Next probe shifted +1mm in world x and +0.5mm in elevation.
    pp[:, 1, 0, 3] = 1.
    pp[:, 1, 2, 3] = .5
    xy, z, fov = project_queries(pp, sp, q, (100, 100), 0, 1)
    assert torch.allclose(xy[0], q.float() * 4 + torch.tensor([-4., 0.]), atol=1e-5)
    assert torch.allclose(z, torch.full_like(z, .5))
    assert bool(fov.all())


def test_feature_matching_shift_and_invalid_candidates():
    b, h, w = 1, 12, 14
    src = torch.zeros(b, 5, h, w)
    dst = torch.zeros_like(src)
    q = torch.tensor([[5, 4], [8, 7]])
    src[0, 0, 4, 5] = 1
    src[0, 1, 7, 8] = 1
    dst[0, 0, 5, 7] = 1
    dst[0, 1, 8, 10] = 1
    pred, _ = match_descriptors(src, dst, q, radius=3)
    assert torch.equal(pred[0], (q + torch.tensor([2, 1])).float() * 4)
    # Radius zero is always exactly the original anchor.
    assert torch.equal(match_descriptors(src, dst, q, radius=0)[0][0], q.float() * 4)
    with pytest.raises(ValueError):
        match_descriptors(src, dst, torch.tensor([[200, 0]]), radius=3)


def test_flow_composition_and_unit_conversion():
    # Each step is exactly +1mm x, with dx=0.5mm/px -> +2 native px each step.
    motion = torch.zeros(1, 4, 3, 12, 12)
    motion[:, :, 0] = 1
    queries = torch.tensor([[4, 4], [8, 8]])
    sp = torch.tensor([[.8, .5]])
    p1 = compose_l1_displacements(motion, queries, sp, 1)
    p4 = compose_l1_displacements(motion, queries, sp, 4)
    assert torch.allclose(p1, queries.float()[None] * 4 + torch.tensor([2., 0.]))
    assert torch.allclose(p4, queries.float()[None] * 4 + torch.tensor([8., 0.]))


def test_raw_patch_grid_and_support_count():
    video = torch.randn(2, 5, 1, 200, 150)
    raw = raw_patch_descriptors(video, (50, 38), patch=9)
    assert raw.shape == (2, 5, 81, 50, 38)
    gt = torch.tensor([[[1., 0.], [5., 0.]]])
    dz = torch.tensor([[0.2, 2.]])
    ok = torch.tensor([[True, True]])
    metric = query_metrics(gt, dz, ok, {'identity': torch.zeros_like(gt)}, max_dz_mm=.5)
    assert metric['proxy_overlap']['n'] == 1
    assert metric['fov_all']['n'] == 2
    assert metric['z_1.5_3']['n'] == 1


def test_query_grid_deterministic():
    a = query_positions((50, 38), 64)
    b = query_positions((50, 38), 64)
    assert torch.equal(a,b)
    assert len(a) <= 64 and int(a[:,0].min()) >= 3
    assert int(a[:,1].min()) >= 3
