import numpy as np

from ct2eco.geometry import pixel_to_probe, probe_to_pixel, rotation, transfer
from ct2eco.render import acoustic_lut, hash_normal


def pose(R=np.eye(3), t=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def test_pixel_probe_roundtrip():
    uv = np.array([[0, 0], [149, 199], [74.5, 10]])
    xyz = pixel_to_probe(uv, (0.4, 0.4), 150)
    assert np.allclose(xyz[2], [0, 4, 0])  # centre column is x = 0
    assert np.allclose(probe_to_pixel(xyz, (0.4, 0.4), 150), uv)


def test_elevational_translation_is_pure_out_of_plane():
    T_s = pose(rotation("y", 30), (10, 20, 30))
    T_t = T_s @ pose(t=(0, 0, 5))  # target plane 5 mm further along the source elevation axis
    p = np.array([[3.0, 40.0, 0.0]])
    assert np.allclose(transfer(T_s, T_t, p), [[3.0, 40.0, -5.0]])
    assert np.allclose(transfer(T_s, T_s, p), p)


def test_tilt_moves_deep_points_further_out_of_plane():
    T_s = pose()
    T_t = pose(rotation("x", 10))  # tilt about the lateral axis, pivoting at the transducer
    z = transfer(T_s, T_t, np.array([[0, 10.0, 0], [0, 80.0, 0]]))[:, 2]
    assert np.allclose(z, [-10 * np.sin(np.deg2rad(10)), -80 * np.sin(np.deg2rad(10))])


def test_rotation_is_proper():
    for ax in "xyz":
        R = rotation(ax, 37)
        assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)


def test_speckle_is_fixed_per_cell_and_standard_normal():
    cells = np.random.default_rng(0).integers(-1000, 1000, (200000, 3))
    a = hash_normal(cells, seed=3, channel=0)
    assert np.array_equal(a, hash_normal(cells, seed=3, channel=0))
    assert abs(a.mean()) < 0.01 and abs(a.std() - 1) < 0.01
    assert abs(np.corrcoef(a, hash_normal(cells, seed=4, channel=0))[0, 1]) < 0.01


def test_acoustic_mapping_follows_sonogym_groups():
    lm = {"background": 0, "body": 1, "liver": 2, "rib_left_3": 3, "kidney_cyst_left": 4, "aorta": 5, "spleen": 6}
    assert acoustic_lut(lm).tolist() == [0, 12, 11, 13, 6, 15, 12]
