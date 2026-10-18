# Local Stream — spatial NCC diagnostic and L2b extension

This incremental patch assumes BOTH prior local-stream patches are applied.
It leaves `USD4RT`, `ct2eco/model.py`, existing checkpoints, and the original
`L0` / `L1` / feature-correlation `L2` networks untouched. It adds **`L2b`**
(spatial B-mode NCC fused into the original L1 dense features), an untrained
NCC calibration baseline and a dense |out-of-plane displacement| probe.
No Global Stream or D4RT decoder is added.

## Exactly what is computed

- `spatial_ncc`: local Pearson NCC between **adjacent native-resolution B-mode
  frames** over odd windows of **15 and 31 pixels**, aggregated to the encoder's
  1/4-resolution grid. Homogeneous/low-variance patches are excluded.
  `ncc` and `ncc_valid` are available from `LocalStream('l2b', ...)`.
- `NCCFusion`: a 3x3 convolution on concatenated 128-channel L1 transition
  features and two NCC-derived decorrelation maps, then a residual 1x1
  projection. The final projection is zero-initialized: with copied L1 encoder
  and pose-head weights, **L2b initially reproduces L1**.
- `dense_abs_z_from_poses`: derives exact **supervision only**, for each source
  image location, from `inv(T_(t+1)) @ T_t` and the sweep's stored `(dy,dx)`
  physical spacing. The network receives the video ONLY; `pose` and `spacing`
  are **never** forwarded to it.
- `DenseDzHead`: a disposable convolutional head predicting a non-negative
  `[B,T-1,1,h,w]` dense map. This evaluates whether features retain *spatial*
  information on out-of-plane displacement; the original global pose-head
  is not a substitute.

**Limit:** L2b's NCC branch uses *zero search radius* (no pre-registration).
For these controlled pure out-of-plane sweeps this isolates the intended
mechanism. On general/mixed motion a lateral alignment step is needed before
interpreting NCC reduction as distance in elevation. NCC also saturates when
frame content becomes uncorrelated. L2b does NOT predict a signed elevation or
full 6-DoF pose from this map.

## 0. Verify installation

```bash
python -m pytest -q tests/test_local_ncc.py tests/test_local_stream.py
```

## 1. Pure image-domain NCC baseline (no GPU training)

Uses existing case-disjoint training and validation CT splits. For each frame
gap, collects deterministic pairs from each sweep, fits **train-only**
monotone decreasing NCC-to-absolute-z-distance calibrators (quantile binning +
weighted isotonic regression), and compares against **train-only spatial
pixel-wise prior**. Also reports error by magnitude bin, mask coverage and
frame gap. `test` is **never** accessed.

```bash
python scripts/eval_local_ncc.py \
  --data data/local_elev_pure --out runs/local/ncc_elev \
  --gaps 1 2 4 --pairs-per-sweep 4
python scripts/eval_local_ncc.py \
  --data data/local_rot_x --out runs/local/ncc_rotx \
  --gaps 1 2 4 --pairs-per-sweep 4
python scripts/eval_local_ncc.py \
  --data data/local_rot_y --out runs/local/ncc_roty \
  --gaps 1 2 4 --pairs-per-sweep 4
```

Outputs `ncc_val.json` (calibrations, `ncc_mae_mm`, `spatial_prior_mae_mm`,
`ratio_vs_spatial_prior`, per-gap counts and magnitude stratification).
A low global MAE caused by predominantly small displacements is NOT evidence
of learning beyond the spatial prior: compare MAE *within each |z| bin*.

## 2. Dense |dz| probe: controlled L1 vs L2b

Use identical data, run length, temporal context, seed and evaluation batches
for all architectures. The head is initialized independently with a fixed
seed offset so added L2b parameters do not affect the probe initialization.
Training uses a Huber loss on |dz| inside **0–3 mm**, and only patches with
reliable local variance. 3 mm is a pre-set *exploratory observability range*,
not a claim of a physical boundary. Validation also reports error over all
valid cells (`full_normal_mae_mm`), to expose censoring effects.

```bash
python scripts/train_local_dense.py \
  --data data/local_elev_pure --motion elevational_pure --arch l1 \
  --frames 5 --steps 1500 --batch 4 --seed 17 \
  --out runs/local/dense_elev_l1_s17
python scripts/train_local_dense.py \
  --data data/local_elev_pure --motion elevational_pure --arch l2b \
  --frames 5 --steps 1500 --batch 4 --seed 17 \
  --out runs/local/dense_elev_l2b_s17

python scripts/train_local_dense.py \
  --data data/local_rot_x --motion rot_x --arch l1 \
  --frames 5 --steps 1500 --batch 4 --seed 17 \
  --out runs/local/dense_rotx_l1_s17
python scripts/train_local_dense.py \
  --data data/local_rot_x --motion rot_x --arch l2b \
  --frames 5 --steps 1500 --batch 4 --seed 17 \
  --out runs/local/dense_rotx_l2b_s17

python scripts/train_local_dense.py \
  --data data/local_rot_y --motion rot_y --arch l1 \
  --frames 5 --steps 1500 --batch 4 --seed 17 \
  --out runs/local/dense_roty_l1_s17
python scripts/train_local_dense.py \
  --data data/local_rot_y --motion rot_y --arch l2b \
  --frames 5 --steps 1500 --batch 4 --seed 17 \
  --out runs/local/dense_roty_l2b_s17
```

Outputs `best.pt`, `last.pt` and `history.jsonl` for each arm. Primary metric:
`normal_mae_mm` on observed `|dz| <=3 mm` (identical selection mask for L1/L2b).
Negative controls: `zeros_mae_mm`, `other_mae_mm`. Additional controls:
`prior_mae_mm` (TRAIN spatial prior) and `full_normal_mae_mm`.

## Interpretation and gates

1. If calibrated NCC alone beats the prior on X/Y, the out-of-plane cue is
   locally measurable; proceed to dense encoder comparisons.
2. If L2b reliably outperforms L1 on new CTs and under input ablation,
   retaining explicit NCC in the local memory is justified.
3. If L1/L2b perform similarly, prefer the simpler L1, and investigate
   correlation quality / window sizes before introducing architectural depth.
4. Do **not** infer signed elevation/rotations, or claim 6-DoF identifiability,
   from an improvement in `|dz|` MAE alone.
5. Repeat promising contrasts on **seeds 42 and 97**, then freeze selection
   before evaluating held-out test CTs with an explicitly separate procedure.

The `scripts/train_local.py --arch l2b` command also works for the original
scalar diagnostic tasks, with existing defaults, but is **not** the dense
feature validation above.
