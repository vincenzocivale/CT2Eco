"""TotalSegmentator subject -> canonical (RAS) CT + one multilabel segmentation, and lookups on it."""

import json
from functools import cached_property
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .geometry import apply

BACKGROUND, BODY = 0, 1


def structure_names(subject_dir):
    return sorted(f.name.removesuffix(".nii.gz") for f in (Path(subject_dir) / "segmentations").glob("*.nii.gz"))


def build_label_map(names):
    """0 = air/outside, 1 = body tissue not covered by any mask, 2.. = TotalSegmentator structures."""
    names = sorted(set(names))
    assert len(names) + 2 <= 256, "labels are stored as uint8"
    return {"background": BACKGROUND, "body": BODY, **{n: i + 2 for i, n in enumerate(names)}}


def body_mask(ct_hu):
    """Largest connected component above -500 HU, holes filled slice by slice (lungs/bowel gas stay inside)."""
    # opening drops 1-voxel sheets, e.g. some TotalSegmentator CTs have a 0 HU padding plane on a face
    m = ndimage.binary_opening(ct_hu > -500, iterations=1)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    m = lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1  # drops the table and other debris
    for k in range(m.shape[2]):
        m[:, :, k] = ndimage.binary_fill_holes(m[:, :, k])
    return m


def _canonical(path):
    img = nib.as_closest_canonical(nib.load(path))
    return np.asarray(img.dataobj), img.affine


def prepare_case(subject_dir, out_dir, label_map):
    """Write out_dir/ct.nii.gz (int16 HU) and out_dir/segmentation.nii.gz (uint8, label_map ids), both RAS."""
    subject_dir, out_dir = Path(subject_dir), Path(out_dir)
    ct, affine = _canonical(subject_dir / "ct.nii.gz")
    seg = np.where(body_mask(ct), BODY, BACKGROUND).astype(np.uint8)
    for name in structure_names(subject_dir):
        if name not in label_map:
            raise KeyError(f"{subject_dir}: structure '{name}' missing from label_map.json")
        m, a = _canonical(subject_dir / "segmentations" / f"{name}.nii.gz")
        if m.shape != seg.shape or not np.allclose(a, affine, atol=1e-3):
            raise ValueError(f"{subject_dir}/{name}: mask grid differs from the CT grid")
        seg[m > 0] = label_map[name]
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(ct.astype(np.int16), affine), out_dir / "ct.nii.gz")
    nib.save(nib.Nifti1Image(seg, affine), out_dir / "segmentation.nii.gz")


class Case:
    """A prepared case: nearest-neighbour label/CT lookups and signed distances in world mm."""

    def __init__(self, case_dir, label_map_path):
        case_dir = Path(case_dir)
        self.id = case_dir.name
        self.dir = case_dir
        self.label_map = json.loads(Path(label_map_path).read_text())
        seg = nib.load(case_dir / "segmentation.nii.gz")
        self.seg = np.asarray(seg.dataobj, np.uint8)
        self.affine = seg.affine
        self.inv_affine = np.linalg.inv(seg.affine)
        self._trees = {}

    @cached_property
    def ct(self):  # only rendering needs it; query-only workers never pay for it
        return np.asarray(nib.load(self.dir / "ct.nii.gz").dataobj, np.int16)

    def _lookup(self, vol, p_world, fill):
        ijk = np.rint(apply(self.inv_affine, p_world)).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < vol.shape), axis=-1)
        out = np.full(p_world.shape[:-1], fill, vol.dtype)
        out[inside] = vol[tuple(ijk[inside].T)]
        return out

    def labels_at(self, p_world):
        return self._lookup(self.seg, p_world, BACKGROUND)

    def ct_at(self, p_world):
        return self._lookup(self.ct, p_world, -1000)

    def inside_volume(self, p_world):
        ijk = np.rint(apply(self.inv_affine, p_world))
        return np.all((ijk >= 0) & (ijk < self.seg.shape), axis=-1)

    def _surface_tree(self, name):
        if name not in self._trees:
            mask = self.seg == self.label_map.get(name, -1)
            tree = None
            if mask.any():
                # shortcut: distance to surface-voxel centres (error <= half a voxel), switch to an EDT if that matters
                surface = mask & ~ndimage.binary_erosion(mask)
                tree = cKDTree(apply(self.affine, np.argwhere(surface).astype(float)))
            self._trees[name] = tree
        return self._trees[name]

    def sdf_at(self, p_world, names):
        """(N, K) signed distance in mm to each named structure: negative inside, NaN if absent in this case."""
        labels = self.labels_at(p_world)
        out = np.full((len(p_world), len(names)), np.nan, np.float32)
        for k, name in enumerate(names):
            tree = self._surface_tree(name)
            if tree is not None:
                d, _ = tree.query(p_world)
                out[:, k] = np.where(labels == self.label_map[name], -d, d)
        return out
