"""Frame conventions and the point-transfer label p_t = T_t^-1 T_s p_s.

Probe frame (mm): x = lateral along the transducer (image columns), y = depth (image rows),
z = elevation = x cross y. The image plane is z = 0, origin at the transducer centre.
Poses T_t are (4, 4) probe -> world (CT RAS mm) matrices.
"""

import numpy as np


def apply(T, p):
    """Apply (..., 4, 4) rigid transforms to (..., 3) points, with broadcasting."""
    return np.einsum("...ij,...j->...i", T[..., :3, :3], p) + T[..., :3, 3]


def pixel_to_probe(uv, spacing, width):
    """(N, 2) pixels (u = column, v = row) -> (N, 3) probe mm on the plane z = 0. spacing = (dy, dx)."""
    uv = np.asarray(uv, float)
    x = (uv[:, 0] - (width - 1) / 2) * spacing[1]
    return np.stack([x, uv[:, 1] * spacing[0], np.zeros_like(x)], 1)


def probe_to_pixel(xyz, spacing, width):
    """Inverse of pixel_to_probe, ignoring z (orthogonal projection on the plane)."""
    return np.stack([xyz[:, 0] / spacing[1] + (width - 1) / 2, xyz[:, 1] / spacing[0]], 1)


def transfer(T_s, T_t, p_s):
    """Express source-probe points in the target-probe frame. Batched over leading dims."""
    return apply(np.linalg.inv(T_t) @ T_s, p_s)


def rotation(axis, deg):
    """Rotation matrix about the probe 'x', 'y' or 'z' axis."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R
