# CT2Eco Local Stream: NCC branch ablation and combined motion

Incremental to `ct2eco-local-ncc-l2b.patch` (and the two older local patches).
Do not regenerate existing pure datasets or re-run existing rot-X/rot-Y probes.
No changes to the original D4RT network, SonoGym renderer, dataset builder,
case splitting, or existing `best.pt` checkpoint format.

## A. Complete the three pure-motion comparisons

These commands work with **the already applied L2b patch**, even without this update:

```bash
python scripts/train_local_dense.py --data data/local_elev_pure --motion elevational_pure \
  --arch l1 --steps 1500 --batch 4 --seed 17 --out runs/local/dense_elev_l1_s17
python scripts/train_local_dense.py --data data/local_elev_pure --motion elevational_pure \
  --arch l2b --steps 1500 --batch 4 --seed 17 --out runs/local/dense_elev_l2b_s17
```

Compare `best.pt` on validation, not arbitrarily the last iteration.

## B. Test whether the NCC branch matters WITHOUT retraining

This patch extends the original dense script with `--eval-only` and
`--ncc-ablation`. It reconstructs L1 + the already trained L2b weights, then
compares identical frames and dense CNN features while varying **only the
NCC-derived decorrelation map**:

- `normal`: correct NCC and original CNN frame maps
- `ncc_zero`: zero NCC evidence fed to the trained fusion (not NCC=0)
- `ncc_other`: map computed on another validation clip, original CNN unchanged
- `ncc_spatial_roll`: decorrelation map shifted in both image axes by half the
  feature grid; original CNN and evaluation support unchanged

There are also the previous image-negative controls (`zeros`, `other`) and
train-fitted spatial prior. No checkpoint weights are updated.

```bash
python scripts/train_local_dense.py --data data/local_rot_x --motion rot_x \
  --arch l2b --batch 4 --frames 5 --val-batches 32 --seed 17 \
  --eval-only --ncc-ablation \
  --init runs/local/dense_rotx_l2b_s17/best.pt \
  --out runs/local/ablation_ncc_rotx_s17

python scripts/train_local_dense.py --data data/local_rot_y --motion rot_y \
  --arch l2b --batch 4 --frames 5 --val-batches 32 --seed 17 \
  --eval-only --ncc-ablation \
  --init runs/local/dense_roty_l2b_s17/best.pt \
  --out runs/local/ablation_ncc_roty_s17

python scripts/train_local_dense.py --data data/local_elev_pure --motion elevational_pure \
  --arch l2b --batch 4 --frames 5 --val-batches 32 --seed 17 \
  --eval-only --ncc-ablation \
  --init runs/local/dense_elev_l2b_s17/best.pt \
  --out runs/local/ablation_ncc_elev_s17
```

Output `evaluation_val.json` with `ncc_zero_mae_mm`, `ncc_other_mae_mm`,
`ncc_spatial_roll_mae_mm`. Compare them to `normal_mae_mm`; absolute
perturbation effect is more useful than `zeros_mae_mm` because *only NCC* is
changed. The negative controls cannot by themselves prove the branch learned a
physical correlation law (fusion can exploit intensity/contrast); check across
CTs/seeds after first screening.

## C. Combined random motion: dense dx, dy, |dz| on the existing pilot

Before training, check `data/pilot_v2/metadata/splits.json` is patient/CT-
disjoint and each sweep has `motion: random` in `sweep.json`. This path is
**not** a pure-motion dataset. If it is missing, use the *already available*
random-motion SonoGym dataset from the previous six-DOF run.

For each location `p=(x,y,0)`, relative pose `G=inv(T_next)@T_current`
produces target `( (Gp)_x-x, (Gp)_y-y, |(Gp)_z| )`. Only the two signed
in-plane components and **unsigned** elevation are supervised. All are in mm.
Only pixels with valid NCC intensity statistics, projected **in-frame**
correspondences, and `|dz|<=3 mm` contribute to optimization and primary MAE.
Report both restricted and unrestricted (valid) MAE and per-component errors.

```bash
python scripts/train_local_dense.py --data data/pilot_v2 --motion random --head xyz \
  --arch l1 --frames 5 --steps 1500 --batch 4 --val-batches 32 --seed 17 \
  --out runs/local/mixed_dense_l1_s17

python scripts/train_local_dense.py --data data/pilot_v2 --motion random --head xyz \
  --arch l2b --frames 5 --steps 1500 --batch 4 --val-batches 32 --seed 17 \
  --out runs/local/mixed_dense_l2b_s17
```

These are **new trainings** with a new 3-channel dense head; neither restores
nor overwrites the old one-channel probes. The NCC branch is still *unregistered*
(zero lateral search), so mixed in-plane displacement may confound speckle
decorrelation. The experiment tests whether that limitation is harmful, not
whether L2b already solves joint pose estimation.

Optional NCC-only ablation after training:

```bash
python scripts/train_local_dense.py --data data/pilot_v2 --motion random --head xyz \
  --arch l2b --frames 5 --batch 4 --val-batches 32 --seed 17 \
  --eval-only --ncc-ablation --init runs/local/mixed_dense_l2b_s17/best.pt \
  --out runs/local/ablation_ncc_mixed_s17
```

In the `xyz` output, `component_names = [dx_mm,dy_mm,abs_dz_mm]`. Their
`component_normal_mae_mm`, `component_prior_mae_mm` and
`component_ncc_zero_mae_mm` fields must be read separately: aggregate MAE can
hide a failure on the elevation component. `evaluated_pixels` counts
**correlated grid cells**, not independent observations; significance must be
assessed by CT case/sweep. No test labels are used by any of these commands.

## Decision

If `ncc_zero/other/spatial_roll` do not significantly deteriorate results,
L2b's advantage over L1 cannot be attributed to its NCC values; prefer L1
or investigate an explicit in-plane registration / correlation volume.
If mixed-motion errors collapse relative to pure tasks, do not add a Global
Stream as a workaround: first verify the in-plane-correspondence geometry and
NCC confounding. Preserve best checkpoints; repeat only the promising arms
on seeds 42 and 97 before a frozen test evaluation.

## Validation

```bash
python -m pytest -q tests/test_local_next.py tests/test_local_ncc.py
```

The local patch verification copy does not include all CT2Eco case/render
modules; CT trajectory tests require running inside the full repository.
