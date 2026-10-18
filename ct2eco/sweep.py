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
    ``lateral`` and legacy ``elevational`` preserve the original re-projection
    of the probe to the skin, which can introduce a depth translation.
    ``elevational_pure`` translates along local z with fixed rotation and an
    initial skin anchor, while ``rot_x/y/z`` rotate around a fixed anchor.
    These latter motions are idealized controlled sweeps, not contact-correct
    acquisitions; they are used only to diagnose identifiability.
    """
    view = str(rng.choice(list(VIEWS)))
    R0 = VIEWS[view]
    centre = target_centre(case, rng, targets) + rng.uniform(-20, 20, 3) * [1, 0, 1]
    elev = random_path(rng, n_frames, rng.uniform(5, 25))  # sweep extent up to 50 mm
    drift = random_path(rng, n_frames, rng.uniform(0, 8))
    angles = {ax: random_path(rng, n_frames, lim) for ax, lim in (("x", 15), ("y", 10), ("z", 10))}
    allowed = {"random", "lateral", "elevational", "elevational_pure", "rot_x", "rot_y", "rot_z"}
    if motion not in allowed:
        raise ValueError(f"Unknown motion mode {motion!r}; expected one of {sorted(allowed)}")
    if motion != "random":
        zero = np.zeros(n_frames)
        elev, drift = zero.copy(), zero.copy()
        angles = {ax: zero.copy() for ax in angles}
        if motion in ("lateral", "elevational", "elevational_pure"):
            line = rng.uniform(-2, 2) * (np.arange(n_frames) - (n_frames - 1) / 2)
            if motion == "lateral":
                drift = line
            else:
                elev = line
        else:
            # degrees per frame, matching ct2eco.geometry.rotation convention
            angles[motion[-1]] = rng.uniform(-1.5, 1.5) * (np.arange(n_frames) - (n_frames - 1) / 2)
    pure = motion in ("elevational_pure", "rot_x", "rot_y", "rot_z")
    anchor = skin_point(case, centre[0], centre[2]) if pure else None
    if pure and anchor is None:
        return None, view
    poses = np.tile(np.eye(4), (n_frames, 1, 1))
    for i in range(n_frames):
        c = centre + R0[:, 2] * elev[i] + R0[:, 0] * drift[i]
        p = anchor + R0[:, 2] * elev[i] + R0[:, 0] * drift[i] if pure else skin_point(case, c[0], c[2])
        if p is None:
            return None, view
        poses[i, :3, :3] = R0 @ rotation("x", angles["x"][i]) @ rotation("y", angles["y"][i]) @ rotation("z", angles["z"][i])
        poses[i, :3, 3] = p
    return poses, view
