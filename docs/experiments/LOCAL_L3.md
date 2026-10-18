# CT2Eco — Frozen L1 + residual elevational NCC (R0/R1/R2)

This patch assumes all earlier CT2Eco Local Stream patches, including
`ct2eco-l1-flow-ncc-diagnostic.patch`, are already applied. It adds *only*
`ct2eco/local_l3_residual.py`, `scripts/train_local_l3.py`,
`tests/test_local_l3.py`, and this guide. It does not change L1, L2b, or D4RT.

## Hypothesis and controls

The **same** existing `mixed_dense_l1_s17/best.pt` provides frozen temporal
feature maps and image-predicted dense `(dx,dy,|dz|)` in millimetres.

- **R0**: frozen L1 features, no NCC input. Controls for head capacity.
- **R1**: frozen L1 features + unregistered multiscale B-mode NCC.
- **R2**: frozen L1 features + unregistered NCC + NCC registered by L1 `(dx,dy)`.

Every variant has the same eight extra feature slots and the same 64/32-channel
residual CNN architecture. Unused slots are zero, and the final layer is zero-
initialized. Consequently all variants reproduce the same L1 predictions
before training. A separate learned branch only adjusts **|dz|** and cannot
change `(dx,dy)`. The original encoder and dense head remain frozen and in
`eval()` throughout training. `softplus` enforces nonnegative |dz|. The L1
motion flow is detached before computing the aligned NCC.

For R2, sample the next B-mode frame at (source pixel + predicted L1 flow),
then compute windowed NCC; invalid warp or texture windows are masked in the
network input, and their validity is provided as a separate signal. All models
use the *same* pixel-validity and label masks for optimization/evaluation;
these masks derive from native fixed NCC validity, source/target CT-pose FOV
and an exploratory <=3mm elevational band. **The pose is only used to build
supervision and the common scoring mask.** It is never passed to the network.

## Apply

```bash
git apply --check ct2eco-local-l3-residual.patch
git apply ct2eco-local-l3-residual.patch
python -m pytest -q tests/test_local_l3.py tests/test_local_next.py tests/test_local_l1_alignment.py
```

## Train (same controls; VAL-only selection)

Run from the CT2Eco repository root:

```bash
for arm in r0 r1 r2; do
  python scripts/train_local_l3.py \
    --data data/pilot_v2 \
    --l1-checkpoint runs/local/mixed_dense_l1_s17/best.pt \
    --variant "$arm" \
    --steps 1500 --batch 4 --val-batches 32 \
    --seed 17 \
    --out "runs/local/l3_${arm}_s17"
done
```

First train R0. If R0 is substantially worse than frozen L1 on validation,
check the objective, masks, and label consistency before comparing NCC arms.
Training uses the same validation windows in every arm (`seed=900001`).

Each run writes `history.jsonl`, `last.pt`, `best.pt`. The only trainable
weights in these checkpoints are the residual's, and the checkpoint records
its frozen L1 SHA-256. The best checkpoint is selected by validation `|dz|`
MAE (<=3mm). Test is not loaded. `full_*` metrics also report the complete
valid region (not restricted to <=3mm). Metrics are reported for dx/dy/dz and
aggregated by CT (few independent validation cases: treat results as pilot).

## Re-evaluate the selected checkpoint without training

```bash
python scripts/train_local_l3.py \
  --data data/pilot_v2 \
  --l1-checkpoint runs/local/mixed_dense_l1_s17/best.pt \
  --variant r2 \
  --batch 4 --val-batches 32 --seed 17 \
  --eval-only --init runs/local/l3_r2_s17/best.pt \
  --out runs/local/l3_r2_eval
```

The evaluator rejects mismatched configs, dataset roots and teacher hashes.

## Decision

Primary: `refined_dz_mae_mm`, plus CT-wise deltas vs frozen L1.
Compare **R2 vs R0 and R1**, paired on the same validation windows, and
ensure it improves frozen L1. A ~5% improvement over R0/R1 is an exploratory
threshold, not a statistical guarantee. `refined_component_mae_mm[0:2]` must
exactly match the frozen baseline's in-plane MAEs across variants.

If R2 does **not** exceed R1 and R0, no need to retain an aligned-NCC branch.
If it does, rerun with seeds 42/97 (same L1 teacher and frozen validation
windows) before selecting a final Local Stream for multi-motion downstreams.
