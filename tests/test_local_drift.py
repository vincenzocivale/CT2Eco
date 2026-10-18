import torch

from scripts.eval_local_drift import triangle_signs


def test_triangle_signs_detect_reversal():
    # signed elevation steps +1, +1, -1, -1, +1 (mm)
    steps = torch.tensor([1., 1., -1., -1., 1.])[:, None]
    gap2 = (steps[:-1] + steps[1:]).abs()
    signs = triangle_signs(steps.abs(), gap2)
    assert torch.equal(signs[:, 0], torch.tensor([1., 1., -1., -1., 1.]))
