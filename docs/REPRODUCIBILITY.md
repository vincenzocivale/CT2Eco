# Reproduction and provenance

Run from the **repository root** in the existing `ct2eco` environment. Existing data and trained checkpoints reside under ignored `data/` and `runs/`. The documentation commit does not bundle proprietary/large medical data, trained weights or generated logs.

## Data, splits, labels

- Main mixed-motion data: `data/pilot_v2`; isolate-motion sets: `data/local_rot_x`, `data/local_rot_y`, `data/local_elev_pure` where previously generated.
- Case-level separation: training CT differ from validation CT and test CT. Reported local experiments used **3 validation CT** (`s0726`, `s1123`, `s1373`). Do not move windows of one CT across splits.
- Read actual split IDs and sweep provenance from `<dataset>/metadata/splits.json` and `sweeps/*/sweep.json`; do not assume that all local datasets share exactly the same patient/sweep counts.
- Ground-truth pose conventions: [`ARCHITECTURE.md`](ARCHITECTURE.md). Train-only calibration; validation-only model selection; **test split untouched** throughout local-model development.

## Primary checkpoints and reports (paths are local, not tracked)

| Output | Role |
|---|---|
| `runs/local/mixed_dense_l1_s17/best.pt` | Frozen L1 checkpoint used by R0/R1/R2 and T1 |
| `runs/local/mixed_dense_l2b_s17/best.pt` | L2b mixed-motion comparator |
| `runs/local/l3_r0_s{17,42,97}/best.pt` | Residual control heads |
| `runs/local/l3_r2_s{17,42,97}/best.pt` | Motion-compensated NCC residual heads |
| `runs/local/l3_seed_confirmation.json` | Three-seed confirmation report |
| `runs/local/l3_inference_cost.json` | Latency/memory report |
| `runs/local/frozen_transfer_t0/frozen_transfer_val.json` | Zero-training correspondence diagnostic |
| `runs/local/t1_{q0,q1,q2}_s17/best.pt` | Query-readout checkpoints |
| `runs/local/t1_transfer_confirmation.json` | Q0/Q1/Q2 gap report |

If checkpoint paths have changed in the actual working tree, pass their true paths explicitly. Never regenerate existing checkpoints unintentionally while preparing a Git commit.

## Minimal verification

```bash
python -m pytest -q tests/test_local_stream.py tests/test_local_ncc.py \
  tests/test_local_next.py tests/test_local_ncc_alignment.py \
  tests/test_local_l1_alignment.py tests/test_local_l3.py \
  tests/test_local_l3_confirmation.py tests/test_local_transfer.py \
  tests/test_local_query_t1.py
```

For a full software release, also run `python -m pytest -q` in the actual repository; the earlier isolated patch-verification environments lacked some original CT helper modules, so their test counts are **not** a validated full-repository test count.

## Reproduce seed comparison and cost (no retraining)

```bash
python scripts/report_local_l3_seeds.py \
  --runs-root runs/local --seeds 17 42 97 \
  --out runs/local/l3_seed_confirmation.json

python scripts/benchmark_local_l3.py \
  --data data/pilot_v2 \
  --l1-checkpoint runs/local/mixed_dense_l1_s17/best.pt \
  --r0 runs/local/l3_r0_s17/best.pt \
  --r2 runs/local/l3_r2_s17/best.pt \
  --batch 4 --warmup 5 --iters 20 \
  --out runs/local/l3_inference_cost.json
```

## Reproduce T0/T1 reports

```bash
python scripts/eval_local_frozen_transfer.py \
  --data data/pilot_v2 \
  --l1-checkpoint runs/local/mixed_dense_l1_s17/best.pt \
  --out runs/local/frozen_transfer_t0 \
  --clips-per-sweep 8 --batch 4 --queries 64 \
  --radius 8 --raw-patch 9 --proxy-max-dz-mm 0.5

python scripts/report_local_query_t1.py \
  --runs-root runs/local --seed 17 \
  --out runs/local/t1_transfer_confirmation.json
```

To retrain Q0/Q1/Q2 under the recorded protocol (1500 steps; this *does* incur GPU compute):

```bash
for arm in q0 q1 q2; do
  python scripts/train_local_query_t1.py \
    --data data/pilot_v2 \
    --l1-checkpoint runs/local/mixed_dense_l1_s17/best.pt \
    --variant "$arm" --steps 1500 --batch 4 --queries 64 \
    --val-clips-per-sweep 8 --eval-every 250 --seed 17 \
    --out "runs/local/t1_${arm}_s17"
done
```

## Reproducibility limitations

- Log-derived summary metrics are preserved in [`RESULTS.md`](RESULTS.md); the full raw per-window predictions, manifest hashes, CT-level confidence intervals and per-CT T1 metrics have not been attached to this commit.
- Training used one/few seeds depending on the gate. The three L3 readout seeds share one frozen L1 model and the same 3 CT validation set.
- The most recent test split results are intentionally absent; do not describe them as completed experiments.
- Model version, code commit, renderer configuration, and device should be captured in future full runs to strengthen provenance.

## L1 v1.1 (2026-10-18)

```bash
python scripts/train_local_dense.py --data data/pilot_v2 --motion random --arch l1 --head xyz \
  --steps 15000 --batch 8 --frames 17 --min-frames 5 --frame-norm --eval-every 1000 --seed 17 \
  --out runs/local/l1v11_fnorm_var_s17
python scripts/eval_local_drift.py --data data/pilot_v2 \
  --l1-checkpoint runs/local/l1v11_fnorm_var_s17/best.pt --mode full \
  --out runs/local/drift/l1v11_fnorm_var_s17_full.json
for arm in q0 q1 q2; do
  python scripts/train_local_query_t1.py --data data/pilot_v2 \
    --l1-checkpoint runs/local/l1v11_fnorm_var_s17/best.pt --variant "$arm" --steps 5000 \
    --batch 4 --queries 64 --val-clips-per-sweep 8 --eval-every 500 --seed 17 \
    --out "runs/local/t1v11_${arm}_s17"
done
```

The training script's own validation uses clips of `--frames` length, so its printed MAE is not comparable across runs with different `--frames`; compare on common support (RESULTS §6).
