# CT2Eco — In-plane compensation / NCC contamination diagnostic

Incremental to the four already-applied local patches, most recently
`ct2eco-local-next.patch`. **No checkpoint changes, no new training, no
modification to the D4RT network, SonoGym renderer or CT split.**

## Question

On `data/pilot_v2` containing `motion=random`, compare NCC in precisely the
same validation pixels under three alternatives:

1. **Fixed NCC:** native B-mode patch correlation at equal source/target pixel coordinates.
2. **Oracle aligned NCC:** warp the target using the exact *in-plane projection*
   of `inv(T_target) @ T_source`. This method **uses ground-truth probe poses**
   and is strictly a diagnostic; it is not necessarily an accuracy upper bound because out-of-plane anatomy changes. Never feed it to an encoder.
3. **Estimated aligned NCC:** use only two B-mode frames to estimate a local
   translation field by coarse NCC patch matching, then warp the target and
   compute native-resolution NCC. This is a cheap *image-only* control, not RAFT.

All conditions use the same windows (15,31 px), the same matching support
(intersection of valid windows and oracle in-plane FOV), and separately fitted
train-only monotone NCC -> |dz| calibration curves. Calibration and spatial
prior are based on **TRAIN CTs only**. VAL CTs are never fitted on.

Results include per-CT, per-abs-dz and per-in-plane-displacement MAE,
paired case bootstrap intervals, image-based matching flow error versus the
oracle in-plane projected flow (diagnostic only), common coverage and values
on the full shared support (including |dz|>3 mm). The primary MAE uses
|dz|<=3mm, as in the dense Local Stream experiments.

**Scientific cautions:** the oracle gives pose-dependent in-plane projections
not necessarily actual observed 2D image correspondence when tissue exits the
ultrasound plane. The image-only local matcher explicitly maximizes NCC, and
its residual NCC can be biased upward even when geometric registration is wrong.
Use both its validation |dz| MAE **and** its in-plane flow EPE. A higher mean
NCC is *not* in itself a success. No result here claims an improvement to the
existing trained L2b head; the experiment tests whether aligned NCC contains
more decodable elevational information before spending budget on L3.

## Run

```bash
python scripts/eval_local_ncc_alignment.py \
  --data data/pilot_v2 \
  --out runs/local/ncc_alignment_mixed \
  --gaps 1 2 4 \
  --train-pairs-per-sweep 4 --val-pairs-per-sweep 8 \
  --windows 15 31 --downsample 4 --match-radius 4 --match-window 7 \
  --max-dz-mm 3 --batch 8 --seed 17
```

For a **smaller first run** use `--gaps 1 --train-pairs-per-sweep 2
--val-pairs-per-sweep 4`. No training takes place.

Check `runs/local/ncc_alignment_mixed/alignment_val.json`:

- `primary_mae_mm` compares fixed, oracle, estimated and train-fitted spatial
  prior on identical valid pixels.
- `macro_ct_mae_mm` and `paired_ct_bootstrap` compare improvements by CT,
  not assuming hundreds of thousands of independent pixels.
- `by_inplane_mm` diagnoses whether misalignment drives the error.
- `by_abs_dz_mm` diagnoses decorrelation saturation at larger distances.
- `flow_epe_px` checks whether the estimated in-plane alignment is credible.
- `coverage` exposes discarded regions, including source patches whose
  projected target is outside FOV.

## Decision rule

If oracle compensation improves decoded |dz| MAE relative to fixed NCC
*on the same validation support*, especially where in-plane movement is
large, pursue explicit correspondence learning; if image-only compensation
also helps and its flow EPE is acceptable, test it directly before training a
new backbone. If only oracle helps, prioritize improving in-plane matching.
If even oracle fails, favor alternative explanations (beam/speckle physics,
rotational effects, an overly simple monotone calibrator) rather than assuming
that further warping or a larger CNN will solve the mixed-motion failure.

Since the oracle **uses poses**, its apparent advantage cannot be interpreted
as a deployable metric; do not use the TEST CTs for method selection.

## Validation

```bash
python -m pytest -q tests/test_local_ncc_alignment.py tests/test_local_ncc.py tests/test_local_next.py
```
