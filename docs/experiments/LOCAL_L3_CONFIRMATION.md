# CT2Eco — L3 confirmation (seeds 17/42/97) and inference cost

This adds reporting/benchmarking scripts only. No changes to training, model
weights, checkpoints, datasets, splits, or test-data handling.

## 1. Confirmation runs

From the root of CT2Eco, keeping **exactly the same frozen L1 checkpoint**:

```bash
for seed in 42 97; do
  for arm in r0 r2; do
    python scripts/train_local_l3.py \
      --data data/pilot_v2 \
      --l1-checkpoint runs/local/mixed_dense_l1_s17/best.pt \
      --variant "$arm" --steps 1500 --batch 4 --val-batches 32 \
      --seed "$seed" \
      --out "runs/local/l3_${arm}_s${seed}"
  done
done
```

The validation window source uses fixed seed `900001`, independently of
`--seed`. Therefore the three training seeds use the same frozen teacher,
case-level split, 128 validation windows and label mask. In each run the
checkpoint is selected by lowest validation <=3 mm elevational MAE. `TEST`
is not accessed.

## 2. Report results

```bash
python scripts/report_local_l3_seeds.py \
  --runs-root runs/local --seeds 17 42 97 \
  --out runs/local/l3_seed_confirmation.json
```

The report validates matched baseline, pixel support and CT keys for R0/R2;
outputs per-seed best epochs, primary and full-support MAE, and paired CT
scores. Repeating seeds does not increase the number of independent CTs.

## 3. Inference latency and GPU memory

Run after training using **one selected pair of checkpoints** (seed 17 is fine
for an initial hardware comparison). Measure the exact forward on identical
five-frame validation clips, excluding data I/O, checkpoint loading, poses,
loss computation and test data:

```bash
python scripts/benchmark_local_l3.py \
  --data data/pilot_v2 \
  --l1-checkpoint runs/local/mixed_dense_l1_s17/best.pt \
  --r0 runs/local/l3_r0_s17/best.pt \
  --r2 runs/local/l3_r2_s17/best.pt \
  --batch 4 --warmup 5 --iters 20 \
  --out runs/local/l3_inference_cost.json
```

`r2_over_r0_median_latency` is the observed forward latency multiplier.
**Caution:** current R0 also computes fixed NCC to construct the shared
validity mask, despite not using NCC values as head input. Therefore the
reported ratio measures extra R2 cost *over the current R0 implementation*
and may underestimate the overhead relative to a fully NCC-free R0. For
publication report hardware, batch, input shape and CUDA/PyTorch versions.

## 4. Engineering decision

Compare R2 vs R0, separately from the guaranteed improvement vs frozen L1.
A >=5% R2 improvement on the primary metric is an exploratory target, not
a significance test. Check the sign of R2-R0 per CT, all-valid error and the
inference time/memory cost. With only three CTs in VAL, do not claim a
statistically precise anatomical-generalization effect. Keep TEST locked
until the model choice is frozen.
