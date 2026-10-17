"""End to end on synthetic phantoms: build -> checks -> Dataset."""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_dataset  # noqa: E402
import check_dataset  # noqa: E402

from ct2eco.queries import QueryDataset  # noqa: E402


def write_phantom(d, shift):
    """180 mm cube at 1.5 mm: soft-tissue slab, a liver sphere and a rib rod, in TotalSegmentator layout."""
    affine = np.diag([1.5, 1.5, 1.5, 1])
    i, j, k = np.mgrid[:120, :120, :120]
    body = (i > 10) & (i < 110) & (j > 20) & (j < 100) & (k > 10) & (k < 110)
    liver = (i - 60 - shift) ** 2 + (j - 70) ** 2 + (k - 60) ** 2 < 20**2
    rib = ((i - 40) ** 2 + (j - 92) ** 2 < 3**2) & body
    ct = np.full(body.shape, -1000, np.int16)
    ct[body], ct[liver], ct[rib] = 40, 60, 700
    (d / "segmentations").mkdir(parents=True)
    nib.save(nib.Nifti1Image(ct, affine), d / "ct.nii.gz")
    for name, m in (("liver", liver), ("rib_left_1", rib), ("spleen", np.zeros_like(body))):
        nib.save(nib.Nifti1Image(m.astype(np.uint8), affine), d / "segmentations" / f"{name}.nii.gz")


def test_pipeline(tmp_path):
    src = tmp_path / "totalseg"
    for n in range(3):
        write_phantom(src / f"s{n:04d}", shift=5 * n)
    out = tmp_path / "ds"
    build_dataset.main(["--totalseg-dir", str(src), "--out", str(out), "--cases", "3", "--sweeps", "2",
                        "--frames", "8", "--queries", "64", "--device", "cpu"])
    assert check_dataset.main([str(out), "--overlays"]) == 0

    item = QueryDataset(out, "train", queries_per_video=32)[0]
    assert item["frames"].shape == (8, 1, 200, 150) and item["frames"].max() > 0
    assert item["queries"]["target_x_mm"].shape == (32,)
    assert item["queries"]["sdf_mm"].shape[0] == 32
    fixed = QueryDataset(out, "val", fixed=True)[0]
    assert len(fixed["queries"]["source_frame"]) == 64
