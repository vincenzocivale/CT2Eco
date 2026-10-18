# CT2Eco — Frozen L1 motion compensation diagnostic

Incremental patch **on top of** `ct2eco-ncc-alignment-diagnostic.patch`.
**No new training**; existing L1 checkpoint is frozen. No change to L1/L2b
architectures or existing scripts.

## Question

On case-disjoint mixed-motion `data/pilot_v2`, does dense L1 prediction of
in-plane displacement (dx,dy) produce a better NCC alignment than classical
block matching? We compare **four** NCC pipelines:

* `fixed`: NCC without registration.
* `oracle`: exact pose-derived in-plane projection. **Diagnostic only**.
* `estimated`: original image-only coarse NCC block matcher.
* `l1`: frozen existing L1 encoder + dense `(dx,dy,|dz|)` head trained on
  mixed sweeps, applying the predicted `(dx,dy)` to register the second frame.

**Important**: L1's trained unit is **mm**. `dx` is divided by `spacing_mm[1]`
(pixel width), `dy` by `spacing_mm[0]` (pixel height) before warping; feature
cells are anchored at image pixel positions `(4*j,4*i)`. No GT poses, GT dz,
or labels enter the L1 prediction. The L1 model takes five consecutive frames,
from which its prediction for the **first adjacent pair** is used. Only
**gap=1** is supported; larger gaps would require composing predicted adjacent
flows and require a separate validation protocol.

Each NCC condition is recalibrated independently using **TRAIN only** (isotonic
NCC->|dz|, same windows and same training support), and is evaluated on exactly
the **same validation pixels**. The common mask includes visibility/FOV and
valid NCC windows of every condition. In-plane flow EPE is measured against
pose-derived projected flow for both image-based methods as a *diagnostic*,
including stratification by out-of-plane distance and in-plane displacement.
Report includes macro-CT MAE and exploratory CT bootstrap (limited number of
validation CTs means broad uncertainty).

## Command

```bash
python scripts/eval_local_l1_alignment.py \
  --data data/pilot_v2 \
  --l1-checkpoint runs/local/mixed_dense_l1_s17/best.pt \
  --out runs/local/ncc_alignment_l1_gap1 \
  --gap 1 --train-pairs-per-sweep 4 --val-pairs-per-sweep 8 \
  --windows 15 31 --downsample 4 --match-radius 4 \
  --match-window 7 --max-dz-mm 3 --batch 8 --seed 17
```

Results: `runs/local/ncc_alignment_l1_gap1/l1_alignment_val.json`.
Check `evaluation.primary_mae_mm`, `evaluation.flow_epe_px`,
`evaluation.by_abs_dz_mm`, `evaluation.by_inplane_mm`, and
`evaluation.by_ct`. These results use the **new four-way support** and different
pair starts from the previous three-way comparison: use the values **within
this file** for strict comparisons, not numbers copied from earlier logs.

## Interpret carefully

* If L1 EPE decreases and its calibrated NCC MAE improves over classical NCC,
  reuse L1 motion heads for an L3 proof of concept before designing RAFT.
* If L1 EPE improves but NCC MAE does not, investigate the decorrelation
  calibration and warping/artifacts; a good 2D flow need not imply better dz.
* If neither improves, consider a confidence-aware cost-volume matcher.
* Oracle uses poses and need not be an attainable or optimal image alignment
  when moving out of plane; it is **not** a deployable result or upper bound.
* Validation uses a small number of CTs. Do not select by pixel counts as if
  millions of correlated pixels were independent; do not access TEST now.

## Tests

```bash
python -m pytest -q tests/test_local_l1_alignment.py \
    tests/test_local_ncc_alignment.py tests/test_local_next.py
```
