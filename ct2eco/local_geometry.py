"""SE(3) point transformation utilities for local pose diagnostics.

All poses map source probe coordinates into next-frame probe coordinates.
Rotations use rotvec XYZ in radians, translations XYZ in mm. No metadata
enters the predictor. Training points and evaluation points are deliberately
different; coordinates are fixed in metric units for reproducibility.
"""

import torch


TRAIN_POINTS_MM = ((-20., 10., 0.), (0., 10., 0.), (20., 10., 0.),
                   (-20., 35., 0.), (0., 35., 0.), (20., 35., 0.),
                   (-20., 60., 0.), (0., 60., 0.), (20., 60., 0.))
EVAL_POINTS_MM = ((0., 0., 0.), (-10., 15., 0.), (10., 15., 0.),
                  (-10., 30., 0.), (10., 30., 0.))


def transform_probe_points(se3, points_mm=TRAIN_POINTS_MM):
    """se3 [...,6] -> transformed points [...,N,3], differentiable."""
    if se3.shape[-1] != 6:
        raise ValueError('Expected rotvec xyz (rad) + translation xyz (mm)')
    w, t = se3[..., :3], se3[..., 3:]
    wx, wy, wz = w.unbind(-1)
    zero = torch.zeros_like(wx)
    skew = torch.stack((zero, -wz, wy,
                        wz, zero, -wx,
                        -wy, wx, zero), -1).reshape(*w.shape[:-1], 3, 3)
    rotation = torch.matrix_exp(skew)
    points = torch.as_tensor(points_mm, dtype=se3.dtype, device=se3.device)
    return torch.einsum('...ij,nj->...ni', rotation, points) + t.unsqueeze(-2)


def point_error_mm(pred, target, points_mm=EVAL_POINTS_MM):
    """Mean Euclidean 3D point error, retaining arbitrary leading dimensions."""
    x = transform_probe_points(pred, points_mm)
    y = transform_probe_points(target, points_mm)
    return (x - y).norm(dim=-1).mean(dim=-1)


def point_loss(pred, target):
    """Differentiable mean Euclidean error on train-only probe-plane points."""
    return point_error_mm(pred, target, TRAIN_POINTS_MM).mean()
