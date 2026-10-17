"""Freehand-like probe trajectories on the anterior skin of a prepared case."""

import numpy as np

from .case import BACKGROUND
from .geometry import rotation

# Probe axes (columns x, y, z) in RAS world. Depth always points posterior.
VIEWS = {
    "transverse": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float).T,  # elevation = inferior
    "longitudinal": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], float).T,  # elevation = right
}
TARGETS = ["liver", "gallbladder", "kidney_right", "kidney_left", "spleen", "pancreas", "stomach", "aorta"]


def skin_point(case, x, z, step=0.5):
    """First non-air voxel met by a ray going posterior from in front of the patient at (x, z)."""
    corners = np.array([[i, j, k] for i in (0, case.seg.shape[0]) for j in (0, case.seg.shape[1]) for k in (0, case.seg.shape[2])])
    y_world = (case.affine[:3, :3] @ corners.T + case.affine[:3, 3:])[1]
    ys = np.arange(y_world.max(), y_world.min(), -step)
    ray = np.stack([np.full_like(ys, x), ys, np.full_like(ys, z)], 1)
    hit = np.flatnonzero(case.labels_at(ray) != BACKGROUND)
    return ray[hit[0]] if len(hit) else None


def target_centre(case, rng, targets=TARGETS):
    """Centroid of a randomly chosen target organ present in the case, else of the body."""
    present = [n for n in targets if n in case.label_map and (case.seg == case.label_map[n]).any()]
    mask = case.seg == case.label_map[rng.choice(present)] if present else case.seg != BACKGROUND
    c = np.argwhere(mask).mean(0)
    return case.affine[:3, :3] @ c + case.affine[:3, 3]


def random_path(rng, n_frames, lim):
    """Piecewise-linear path through 2-5 random knots in [-lim, lim]: random direction, speed, stops and reversals."""
    knots = rng.integers(2, 6)
    return np.interp(np.arange(n_frames), np.linspace(0, n_frames - 1, knots), rng.uniform(-lim, lim, knots))


def make_trajectory(case, rng, n_frames, targets=TARGETS, motion="random"):
    """((n_frames, 4, 4) probe->world poses, view name); poses is None if the probe leaves the skin.

    Elevation, lateral drift, tilt/yaw/rock all follow independent random paths, so the frame gap carries no
    information about the motion: the model has to read it from the images.
    motion="lateral" / "elevational": sanity datasets, one pure translation at a constant random speed per sweep
    (-2..2 mm/frame), no rotation. The speed is only observable from the images.
    """
    view = str(rng.choice(list(VIEWS)))
    R0 = VIEWS[view]
    centre = target_centre(case, rng, targets) + rng.uniform(-20, 20, 3) * [1, 0, 1]
    elev = random_path(rng, n_frames, rng.uniform(5, 25))  # sweep extent up to 50 mm
    drift = random_path(rng, n_frames, rng.uniform(0, 8))
    angles = {ax: random_path(rng, n_frames, lim) for ax, lim in (("x", 15), ("y", 10), ("z", 10))}
    if motion != "random":
        line = rng.uniform(-2, 2) * (np.arange(n_frames) - (n_frames - 1) / 2)
        zero = np.zeros(n_frames)
        elev, drift = (zero, line) if motion == "lateral" else (line, zero)
        angles = {ax: zero for ax in angles}
    poses = np.tile(np.eye(4), (n_frames, 1, 1))
    for i in range(n_frames):
        c = centre + R0[:, 2] * elev[i] + R0[:, 0] * drift[i]
        p = skin_point(case, c[0], c[2])
        if p is None:
            return None, view
        poses[i, :3, :3] = R0 @ rotation("x", angles["x"][i]) @ rotation("y", angles["y"][i]) @ rotation("z", angles["z"][i])
        poses[i, :3, 3] = p
    return poses, view
