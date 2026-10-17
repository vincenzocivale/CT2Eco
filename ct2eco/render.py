"""Render a sweep: slice the case along the poses and run the SonoGym convolution simulator.

Speckle comes from deterministic noise attached to 3D world cells, so the same tissue gives the same
scatterers in every frame of a sweep; a new seed (new sweep) draws new scatterers and acoustic jitter.
"""

import copy
import re
from pathlib import Path

import numpy as np
import torch
import yaml

from .geometry import apply, pixel_to_probe
from .sonogym_conv import USSimulatorConv

SONOGYM_CFG = yaml.safe_load((Path(__file__).parent / "sonogym_us_cfg.yaml").read_text())
for _p in SONOGYM_CFG["label_to_ac_params_dict"].values():
    _p.update({k: float(v) for k, v in _p.items()})  # YAML 1.1 reads '0.04e5' as a string
CT_RANGE = (-300, 3000)  # SonoGym scene cfg `ct_range`, mapped to 0..255

# TotalSegmentator name -> SonoGym acoustic class, same grouping as SonoGym's label_conversion.yaml
ACOUSTIC_RULES = [
    (r"lung", 2),
    (r"kidney", 6),
    (r"liver", 11),
    (r"skull", 14),
    (r"vertebrae|sacrum|rib|femur|hip|humerus|scapula|clavicula|sternum|costal", 13),
    (r"heart|gluteus|autochthon|iliopsoas", 8),
    (r"aorta|arter|vein|vena|atrial|brachiocephalic|portal", 15),
    (r"spinal_cord", 10),
    (r"brain", 9),
]
SOFT_TISSUE = 12  # SonoGym maps the body and every other organ here


def acoustic_lut(label_map):
    lut = np.full(max(label_map.values()) + 1, SOFT_TISSUE, np.int64)
    lut[label_map["background"]] = 0
    for name, idx in label_map.items():
        lut[idx] = next((cls for pat, cls in ACOUSTIC_RULES if re.search(pat, name)), lut[idx])
    return lut


def _splitmix(z):
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def hash_normal(cells, seed, channel):
    """Deterministic N(0, 1) value per integer 3D cell (..., 3)."""
    c = np.ascontiguousarray(cells, np.int64).view(np.uint64)
    key = c[..., 0] * np.uint64(73856093) ^ c[..., 1] * np.uint64(19349663) ^ c[..., 2] * np.uint64(83492791)
    key = _splitmix(key ^ np.uint64((seed * 16 + channel) % 2**64))
    u1 = ((key >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0**-53
    u2 = (_splitmix(key) >> np.uint64(11)).astype(np.float64) * 2.0**-53
    return (np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)).astype(np.float32)


def jittered_cfg(rng, scale=0.2):
    """SonoGym parameters with per-sweep multiplicative jitter on attenuation and scatterer intensity."""
    cfg = copy.deepcopy(SONOGYM_CFG)
    for params in cfg["label_to_ac_params_dict"].values():
        params["alpha"] *= rng.uniform(1 - scale, 1 + scale)
        params["mu0"] *= rng.uniform(1 - scale, 1 + scale)
    return cfg


def beam_profile(fwhm_mm, n_slices):
    """Elevational offsets (mm) and Gaussian weights; fwhm 0 = infinitely thin plane (SonoGym's 2D default)."""
    if fwhm_mm <= 0:
        return np.zeros(1), np.ones(1)
    offsets = np.linspace(-fwhm_mm, fwhm_mm, n_slices)  # +-1 FWHM covers the beam down to ~6% intensity
    w = np.exp(-0.5 * (offsets / (fwhm_mm / 2.3548)) ** 2)
    return offsets, w / w.sum()


def render_sweep(case, poses, image_size, spacing, seed, use_ct=True, device="cpu", beam_fwhm_mm=0.0, beam_slices=7):
    """-> (frames uint8 (n, H, W), labels uint8 (n, H, W) on the central plane) for probe->world poses (n, 4, 4).

    beam_fwhm_mm > 0 integrates the image over the elevational beam: the simulator is linear in the scatterer
    field, so the weighted sum of sub-slice images equals integrating the scatterers across the beam before the PSF.
    Overlapping beams give the gradual speckle decorrelation that encodes how far the plane moved.
    """
    H, W = image_size
    v, u = np.mgrid[:H, :W]
    p_plane = pixel_to_probe(np.stack([u.ravel(), v.ravel()], 1), spacing, W)
    cfg = jittered_cfg(np.random.default_rng(seed))
    sim = USSimulatorConv(cfg, device)
    lut = acoustic_lut(case.label_map)
    # shortcut: one large-scale speckle size for all tissues (SonoGym's cfg uses fl = 30 everywhere), per-label if that changes
    fl = max(p["fl"] for p in cfg["label_to_ac_params_dict"].values())
    cell = min(spacing)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)

    labels = case.labels_at(apply(poses[:, None], p_plane[None]).reshape(len(poses), H, W, 3))  # central plane
    us = 0.0
    for offset, weight in zip(*beam_profile(beam_fwhm_mm, beam_slices)):
        p_world = apply(poses[:, None], (p_plane + [0, 0, offset])[None]).reshape(len(poses), H, W, 3)
        lab = labels if offset == 0 else case.labels_at(p_world)
        T0 = hash_normal(np.floor(p_world / cell), seed, 0)
        T1 = hash_normal(np.floor(p_world / cell), seed, 1)
        Vl = hash_normal(np.floor(p_world / (fl * cell)), seed, 2)
        ct = (np.clip(case.ct_at(p_world), *CT_RANGE) - CT_RANGE[0]) / (CT_RANGE[1] - CT_RANGE[0]) * 255
        ct = np.floor(ct)  # SonoGym slices the CT into an int32 tensor
        with torch.no_grad():
            us = us + weight * sim.simulate_US_image_given_rand_map(
                t(lut[lab]), t(T0), t(T1), t(Vl), if_ct=use_ct, ct_img=t(ct)
            ).cpu().numpy()
    # one scale for the whole sweep: per-frame normalisation would make brightness flicker with the content
    us = np.clip(us / max(np.percentile(us, 99.5), 1e-8), 0, 1)
    return (us * 255).astype(np.uint8), labels
