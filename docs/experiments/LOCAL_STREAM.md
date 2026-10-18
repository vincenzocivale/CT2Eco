# CT2Eco Local Stream — isolated feasibility protocol

**Scope:** this module is an independent diagnostic experiment, not a replacement for
`USD4RT` in `ct2eco/model.py`. Nothing is modified in the existing D4RT training
pipeline. Pose ground truth is used **only for the disposable diagnostic head**.
Neither GT poses nor frame indices, sweep IDs or CT labels enter the encoder.

## Architecture (implemented)

| Arm | Visual input | Dense representation | Diagnostic head |
|---|---|---|---|
| L0 | Two adjacent frames, early-fused 2-D residual CNN | `[B,T-1,128,H/4,W/4]` | 4×4 spatial pooling + MLP |
| L1 | Local 3-D residual CNN with temporal stride 1 | Transition maps as above; additionally `[B,T,128,H/4,W/4]` frame maps | Same MLP |
| L2 | L1 + 9×9 normalized correlation of neighbouring-frame feature maps | Same as L1 | Same MLP |

The *full dense feature maps* are available for subsequent query decoding. The
small regression head is a temporary diagnostic, not the eventual D4RT decoder.
There is **no spatial/global attention** in this phase (future L3 is conditional
on evidence that L0/L1/L2 can actually learn motion).

The target reference convention follows `ct2eco/queries.py`:
`T_(t+1)^(-1) @ T_t`, mapping a point from frame `t` into the coordinates of
frame `t+1`. This gives signed local x translation, signed in-plane z rotation,
elevation translation magnitude, or full 6-DoF relative motion.

## Run

All commands below must run **from the CT2Eco repository root** after applying
the patch, in the existing CT2Eco Python environment. No new required packages:
`torch`, `numpy`, `scipy`, `pytest` are already used in CT2Eco.

```bash
python -m pytest -q tests/test_local_stream.py

# Gate 0: same phantom, opposite velocities; only images distinguish them
python scripts/train_local.py --source phantom --task lateral \
    --arch l0 --batch 2 --steps 1500 --overfit-opposites \
    --out runs/local/opposites_l0_s17
python scripts/train_local.py --source phantom --task lateral \
    --arch l1 --batch 2 --steps 1500 --overfit-opposites \
    --out runs/local/opposites_l1_s17

# Gate 1: observable movement in independent online 3-D speckle phantoms
python scripts/train_local.py --source phantom --task lateral \
    --arch l0 --steps 1500 --out runs/local/lateral_l0_s17 --seed 17
python scripts/train_local.py --source phantom --task lateral \
    --arch l1 --steps 1500 --out runs/local/lateral_l1_s17 --seed 17

# Only if L1 passes: explicit local correlation ablation
python scripts/train_local.py --source phantom --task lateral \
    --arch l2 --steps 1500 --out runs/local/lateral_l2_s17 --seed 17

# Gate 2: in-plane rotation and sign-free elevational displacement
python scripts/train_local.py --source phantom --task inplane_rot \
    --arch l1 --steps 1500 --out runs/local/rot_l1_s17
python scripts/train_local.py --source phantom --task elev_mag \
    --arch l1 --steps 1500 --out runs/local/elev_l1_s17

# Gate 3: composite SE(3) (6-vector = rotvec rad and translation mm)
python scripts/train_local.py --source phantom --task rigid \
    --arch l1 --steps 1500 --out runs/local/rigid_phantom_s17

# Then use actual CT2Eco / SonoGym generated videos with case-disjoint split.
# NB: existing random sweeps mix motion axes; choose --task rigid here.
python scripts/train_local.py --source ct2eco --data data/pilot_v2 \
    --task rigid --arch l1 --frames 5 --steps 3000 --batch 4 \
    --out runs/local/rigid_ct_s17
```

To run sensitivity seeds, repeat the selected experiments using `--seed 42`
and `--seed 97`. Use **matching compute budgets and data** for comparisons.
For CT2Eco runs the script uses `metadata/splits.json` to separate train and
validation CT cases; it samples only contiguous windows, supervises adjacent
pairs, and never passes `poses` to the model. Additional geometry/per-case
stress tests on *separate CT volumes* should follow before publication.

To test the learned dense representation without updating the backbone:

```bash
python scripts/train_local.py --source ct2eco --data data/pilot_v2 \
  --task rigid --arch l1 --init runs/local/rigid_ct_s17/best.pt \
  --freeze-encoder --out runs/local/rigid_ct_headonly_s17
```

This is head-only adaptation on the same task, *not* yet a test of transfer to
a different D4RT query task.

## What we measure

At every evaluation we save metrics in `history.jsonl` and `best.pt`/
`last.pt` checkpoints:

- `normal_mae`: target-component MAE in physical task units for scalar tasks
  (mm or radians); for `rigid`, mean error over six components normalized by
  fixed scales (0.04 rad/axis and 0.8 mm/axis). Per-component errors and
  3-D five-point registration error remain in physical units.
- `blind_mae`: image-free target-mean baseline estimated from **train only**.
- `ratio_vs_blind`: the model must beat this baseline to justify more training.
- `zeros_mae`, `other_video_mae`, `time_shuffle_mae`: identical targets but null/incorrect videos,
  to detect a model using motion priors rather than images.
- `time_reverse_mae`: forward-reverse consistency diagnostic, including exact
  SE(3) inversion for the rigid head (this is NOT ground-truth error).
- `sign_accuracy` (lateral and in-plane rotation), `per_component_mae`, and
  3-D five-point registration error (`point_mae_mm`) for the rigid head.

**Initial feasibility gates (proposed, not demonstrated):**

1. Overfit *two videos synthesized from the same 3-D phantom with opposite motion*
   when coordinates/time are identical (`--overfit-opposites`). This is a fit-only
   gate; afterwards generalize to new randomized phantoms.
2. Achieve `normal_mae < 0.7 * blind_mae`, with worsening error on zeroed or
   cross-video inputs. Thresholds are screening criteria, not benchmark claims.
3. Test the same motion with reversed frame order and check SE(3) consistency.
4. Only then test mixed motion on CT2Eco videos and compare L0, L1 and L2 using
   independent CT cases and three seeds.

The phantom validation set uses a fixed random seed across architectures and
training seeds; CT2Eco validation windows are likewise fixed. The online `phantom` source is a controlled 3-D noise/scatterer field sampled
with analytic poses; it is **not** a realistic ultrasound simulator and does not
replace the SonoGym/CT pipeline. It is used to isolate network learnability
from ultrasound simulator details. In particular, the elevational problem
may be fundamentally sign-ambiguous.

**Important:** the existing `CT2Eco/EXPERIMENTS.md` documents a prior failure
of the complete USD4RT network even on a simple lateral-motion sanity check.
Passing a shape/gradient unit test is NOT evidence of motion learning. Evaluate
image ablations and blind priors before interpreting any decrease in loss.

## Future integration with US-D4RT

`LocalStream(...).forward(video)` returns `dense` per transition and, for L1/L2,
`frame_dense` per frame; preserve these to add local motion information to the
memory accessed by the D4RT query decoder. A global stream and query decoder
can be added **after** the isolated motion gate. `ct2eco/model.py` and
`scripts/train.py` remain unchanged by this patch.

## Follow-up patch — 6-DoF diagnostics, pure-axis SonoGym sweeps, point objective

This section extends the original Local Stream patch. All **original defaults
remain unchanged** (`--loss huber`, validation selection by `normal_mae`,
existing model weights/state_dict names preserved). It does not modify USD4RT.

### Re-evaluate the existing 6-DoF checkpoint without retraining

```bash
python scripts/train_local.py \
  --source ct2eco --data data/pilot_v2 --task rigid --arch l1 \
  --frames 5 --batch 4 --seed 17 --eval-only \
  --init runs/local/rigid_ct_l1_s17/best.pt \
  --out runs/local/rigid_ct_l1_s17_reanalysis
```

`evaluation_val.json` and the terminal output include:

- `per_component_normal_mae`, `per_component_zeros_mae`,
  `per_component_other_video_mae`, and `per_component_time_shuffle_mae` in
  raw units: rotvec components [rad] and translations [mm].
- `per_component_blind_mae` from the **training-only target mean**.
- `point_normal_mae_mm` (`point_mae_mm` alias), `point_zeros_mae_mm`,
  `point_other_video_mae_mm`, `point_time_shuffle_mae_mm`,
  `point_blind_mae_mm` and `point_identity_mae_mm`.
- `time_reverse_mae` remains a prediction consistency test, **not** the error
  to ground truth on reversed sequences.

Five physically fixed **evaluation** points are used for all 3D point errors.
The identity baseline predicts zero motion, the blind baseline uses a training
mean rotvec+translation. Input corruption keeps the original target on purpose.

`--eval-split test` is available **only with `--eval-only --init`**. Use the
held-out test data exactly once after freezing your model selection; the
validation set remains appropriate for these exploratory experiments.

### Isolated CT-based movement: what counts as "pure"

The legacy `--motion elevational` path reprojects every frame back onto the
skin and can introduce **additional depth displacement**. It is not strictly
a pure-elevation SE(3) trajectory. The new `elevational_pure` mode fixes a
single probe anchor and translates **only along local z**. The new `rot_x`,
`rot_y` and `rot_z` modes rotate about that fixed anchor. These are idealized
kinematics for signal-identifiability controls; they do **not** model changing
skin contact accurately and should not replace the production dataset.

Regenerate dedicated datasets (never reuse the output directory of a previous
motion mode because `build_dataset.py` skips existing completed sweeps):

```bash
# Supply your actual TotalSegmentator source directory.
TOTAL_SEG=/path/to/TotalSegmentator_subjects

python scripts/build_dataset.py --totalseg-dir "$TOTAL_SEG" \
  --out data/local_elev_pure --motion elevational_pure \
  --cases 20 --sweeps 8 --frames 32 --beam-fwhm-mm 2.0

python scripts/build_dataset.py --totalseg-dir "$TOTAL_SEG" \
  --out data/local_rotx --motion rot_x \
  --cases 20 --sweeps 8 --frames 32 --beam-fwhm-mm 2.0

python scripts/build_dataset.py --totalseg-dir "$TOTAL_SEG" \
  --out data/local_roty --motion rot_y \
  --cases 20 --sweeps 8 --frames 32 --beam-fwhm-mm 2.0
```

If GPU support or clinical CT data are unavailable, the corresponding task
controls can first run on `--source phantom`; success on a phantom is NOT
proof of success on the SonoGym renderer. Keep splits CT-disjoint.

Train independent single-target heads on the pure datasets:

```bash
python scripts/train_local.py --source ct2eco --data data/local_elev_pure \
  --task elev_mag --arch l1 --frames 5 --steps 3000 --seed 17 \
  --require-pure-motion --out runs/local/elev_mag_ct_s17

python scripts/train_local.py --source ct2eco --data data/local_elev_pure \
  --task elev_signed --arch l1 --frames 5 --steps 3000 --seed 17 \
  --require-pure-motion --out runs/local/elev_signed_ct_s17

python scripts/train_local.py --source ct2eco --data data/local_rotx \
  --task rot_x --arch l1 --frames 5 --steps 3000 --seed 17 \
  --require-pure-motion --out runs/local/rotx_ct_s17

python scripts/train_local.py --source ct2eco --data data/local_roty \
  --task rot_y --arch l1 --frames 5 --steps 3000 --seed 17 \
  --require-pure-motion --out runs/local/roty_ct_s17
```

New tasks: `elev_signed` predicts translation z with sign; `rot_x` and `rot_y`
predict the X/Y components of the **SO(3) rotation vector** (radians).
`--require-pure-motion` rejects a dataset with incompatible `sweep.json`
`motion` metadata. Without it, a single-target run on `pilot_v2` estimates a
component in *mixed* motion — a useful diagnostic, but **not** a pure-motion
experiment. Quantify all values on CT-held-out validation cases.

### Controlled change of 6-DoF objective: point geometry

Training uses nine fixed 3D points within the probe imaging plane; evaluation
uses a **different** five-point panel. Rotations use exact SO(3) exponential
(`torch.matrix_exp`, differentiable). The new objective is the mean 3D Euclidean
point error in millimetres; pose annotations remain training targets, never
model inputs. The baseline arm still uses the component-wise Huber objective.

```bash
# Reproduce the baseline at equal compute budget, seed, and data:
python scripts/train_local.py --source ct2eco --data data/pilot_v2 \
  --task rigid --arch l1 --frames 5 --steps 3000 --batch 4 \
  --seed 17 --loss huber --select-metric point_mae_mm \
  --out runs/local/rigid_huber_pointselect_s17

# Single changed factor: optimization objective
python scripts/train_local.py --source ct2eco --data data/pilot_v2 \
  --task rigid --arch l1 --frames 5 --steps 3000 --batch 4 \
  --seed 17 --loss point --select-metric point_mae_mm \
  --out runs/local/rigid_point_pointselect_s17
```

Select checkpoints on **the same validation metric** (`point_mae_mm`) for both
arms. Compare the raw component MAEs and 3D point errors, including the
identity and image-free baselines. The test split must remain locked until
architecture and checkpoint choices are finalized. Repeat a successful
comparison with seeds 42 and 97. Do not interpret improvement on one seed or
one 3-CT validation split as publishable confirmation.

## Spatial NCC / L2b follow-up

See [LOCAL_NCC.md](LOCAL_NCC.md) for train-only NCC calibration, the new
`--arch l2b` feature fusion and dense `|dz(x,y)|` probe comparison against L1.
Existing Local Stream experiments and checkpoints remain compatible.
