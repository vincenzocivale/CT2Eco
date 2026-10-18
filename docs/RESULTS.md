# Experimental results — through 2026-10-17

**Source:** user-provided training/evaluation logs, validated only on CT-disjoint **VAL**, with **TEST untouched**. Numeric precision reflects the printed logs, not inferential certainty. Each panel describes its own evaluation support; **do not cross-compare absolute MAEs from different panels as if they shared identical queries and masks**. The earlier original US-D4RT results are in [`../EXPERIMENTS.md`](../EXPERIMENTS.md).

## 1. Local Stream learning on pure movements

Best-validation dense `|dz|` MAE (mm), 1500-step L1/L2b probes (seed 17), separate pure-motion datasets and masks:

| Motion | Spatial prior | L1 | L2b | L2b relative improvement vs L1 |
|---|---:|---:|---:|---:|
| Rotation X | 0.24554 | 0.11203 | **0.10528** | 6.0% |
| Rotation Y | 0.09392 | 0.07725 | **0.06869** | 11.1% |
| Pure elevation | 0.44382 | 0.19916 | **0.19330** | 2.9% |

For rot-X and rot-Y, the dense target has a strong low-dimensional rigid pattern; success alone does **not** demonstrate local anatomical matching. For signed pure elevation/rotations, previous isolated probes did not demonstrate robust sign recovery. A pure elevation-magnitude run did recover useful magnitude signal with perturbed-image controls.

## 2. Combined motion (random, `data/pilot_v2`)

Best-validation results, seed 17, 777,762 primary pixel observations (840,076 full valid):

| Model | Mean of 3 components | `dx` MAE | `dy` MAE | `|dz|` MAE |
|---|---:|---:|---:|---:|
| Spatial prior | 0.45108 | 0.38484 | 0.38932 | 0.57909 |
| **L1** (best at step 1350) | **0.30606** | **0.26351** | **0.26200** | **0.39267** |
| L2b (step 1500) | 0.39584 | 0.39408 | 0.36353 | 0.42993 |

Both are image-dependent under perturbation. L2b uses NCC for elevation (ablation `|dz|` MAE: **0.42993 original**, **0.74455 NCC zero**, **0.61905 NCC from another video**, **0.47775 NCC spatial roll**) but harms the joint readout relative to L1. Perturbation is not an independent causal proof of representational transfer.

## 3. Motion compensation diagnostics (different support from §2)

Train-only monotone NCC → `|dz|` calibration, validation-only results and common support within each gap:

| Gap | Fixed NCC | Oracle warp | Classical estimated warp | Common coverage | Classical flow EPE |
|---|---:|---:|---:|---:|---:|
| 1 | 0.5053 | **0.4641** | 0.5154 | 76.7% | 2.06 px |
| 2 | 0.6210 | **0.5385** | 0.6048 | 73.0% | 2.82 px |
| 4 | 0.6941 | **0.6168** | 0.6935 | 68.8% | 3.68 px |

At gap 1, a new four-way **common-support** comparison with 3 validation CT and 192 frame pairs gave:

| Approach | `|dz|` calibration MAE (mm) | 2-D flow EPE |
|---|---:|---:|
| Fixed | 0.5063 | — |
| Classical matching | 0.5040 | 1.936 px |
| **L1-predicted flow warp** | **0.4879** | **0.970 px** |
| Oracle warp (GT diagnostic only) | **0.4546** | 0 by construction on geometric projection |
| Train-only prior | 0.5822 | — |

The L1 warp reduces NCC error modestly; oracle remains better. The two tables use **different support**, so compare differences only within a table. Warp with oracle flow is not a usable image-only inference system.

## 4. Frozen L1 + residual elevation, three seeds

`R0` vs motion-compensated `R2`: identical frozen L1 checkpoint, matched architecture/head capacity, seeds **17, 42, 97**; selected by validation `|dz|` MAE.

| Seed | R0 MAE (mm) | R2 MAE (mm) | Relative reduction |
|---|---:|---:|---:|
| 17 | 0.35562 | **0.34208** | 3.81% |
| 42 | 0.35277 | **0.33804** | 4.18% |
| 97 | 0.35776 | **0.33848** | 5.39% |
| Mean | 0.35538 | **0.33953** | 4.46% (mean per seed) |

On seed 17, R1 (fixed NCC) achieved **0.34903**; R0/R1/R2 improvement is partially due to the added residual head (R0 itself improves L1 from 0.39267 to 0.35562). R2 was better across all **three seed aggregates**, but **not on every CT × seed combination**. Only 3 validation CT are independent; no generalization confidence interval is established.

Forward-only implementation-specific benchmark (CUDA, batch 4, 5 × 200 × 150 frames, 20 iterations, excluding I/O): R0 **4.998 ms/batch**, R2 **5.792 ms/batch** (**+15.9%**); allocated peak about **129 MiB** for both. R0 implementation still computes fixed NCC for validity; optimized overhead could differ. Reports: `runs/local/l3_seed_confirmation.json`, `runs/local/l3_inference_cost.json` (not committed).

## 5. Frozen Local representation probes

### T0 — zero-training sparse matching

EPE (px) on the exploratory overlap proxy `|dz| ≤ 0.5 mm`; evaluation used 3 CT/24 sweeps and 192 clips per gap. The count of proxy queries decreases with gap (3,932 / 1,687 / 812).

| Method | Gap 1 | Gap 2 | Gap 4 |
|---|---:|---:|---:|
| Identity | 1.46 | 2.86 | 4.79 |
| Raw B-mode matching | 12.63 | 14.91 | 16.51 |
| Random L1 cosine | 16.56 | 21.51 | 26.11 |
| Trained L1 cosine | 18.26 | 22.23 | 24.88 |
| **Previously supervised L1 motion head** | **0.85** | **1.74** | **2.81** |

The direct cosine descriptor matching **fails**, even against identity. This does not establish that frozen transition representations lack usable information, and the supervised motion head is not a transfer-only control. Ground-truth projections in small `|dz|` strata are still not guaranteed acoustically visible correspondences.

### T1 — train a new query readout, freeze L1

Validation EPE (px), gap 1/2/4, 3 held-out validation CT:

| Gap | Q0 coordinates | Q1 random L1 | **Q2 pretrained L1** | Q2 reduction vs Q1 |
|---|---:|---:|---:|---:|
| 1 | 1.675 | 1.672 | **1.344** | 19.6% |
| 2 | 3.092 | 3.085 | **2.194** | 28.9% |
| 4 | 5.858 | 5.851 | **4.520** | 22.7% |

**Gate T1 passed on aggregate UV EPE.** Q2 outperforms both controls on all three gaps without encoder updates. This demonstrates learnable geometry readout from a pretrained frozen encoder on the **same synthetic task family**, not independent task/domain transfer, persistent point identity, or full 3-D signed pose recovery. The per-CT and `|z'|` T1 results were **not supplied** in the reported summary; retrieve them from `runs/local/t1_transfer_confirmation.json` before citing those outcomes.

## 6. L1 v1.1 consolidation (2026-10-17/18)

Eval-only diagnostics on the original checkpoint, then retraining. All on **VAL** (3 CT, 24 sweeps), seed 17 only, test untouched. MAE = mean of dense `dx, dy, |dz|` (mm), mask NCC-valid ∩ in-FOV ∩ `|dz| ≤ 3 mm`, over **all** adjacent transitions of the 24 sweeps (larger support than §2).

**Diagnosis of `mixed_dense_l1_s17`.** Train MAE 0.310 ≈ val 0.308 (no overfitting) while the val curve was still falling steeply when the cosine LR reached 10% at step 1500 → undertrained. Accuracy also degraded when run on clips longer than the 5 frames used in training (GroupNorm statistics mixed across time).

| L1 run on | Original (1.5k steps) | Long training (15k, batch 8) | **Per-frame GroupNorm + clips 5–17 (15k, batch 8)** |
|---|---:|---:|---:|
| 5-frame windows | 0.308 | 0.165 | **0.163** |
| T=9 clips (interior) | 0.310 | 0.174 | **0.158** |
| T=17 clips (interior) | 0.327 | 0.204 | **0.163** |
| full 32-frame sweep (interior) | 0.366 | 0.241 | **0.160** |

Full-sweep components for the new model: `dx` 0.114, `dy` 0.127, `|dz|` 0.239 mm. Zeroed-frame control ≈ 0.60 mm (image-dependent). Checkpoint: `runs/local/l1v11_fnorm_var_s17/best.pt`.

**Sweep-level drift** (`scripts/eval_local_drift.py`, TUS-REC-inspired). Per transition, a Kabsch rigid fit of the dense prediction, chained over the sweep; mean image-grid point error in world mm. GPE = mean over frames, FDE = last frame, LPE = per transition. Mean probe path 33.6 mm per sweep.

| Model | Variant | GPE | FDE (% path) | LPE |
|---|---|---:|---:|---:|
| — | identity (static probe) | 11.56 | 17.27 (63.6%) | 1.416 |
| original L1 | in-plane only (image-only) | 10.37 | 15.37 (55.4%) | 1.235 |
| new L1, full sweep | in-plane only (image-only) | 9.64 | 14.32 (51.5%) | 1.140 |
| original L1 | **oracle dz sign** (diagnostic) | 5.09 | 7.42 (24.9%) | 0.637 |
| new L1, full sweep | **oracle dz sign** (diagnostic) | **2.95** | **4.41 (15.1%)** | **0.342** |

Without the elevation sign, image-only drift is close to the static-probe baseline: **the missing dz sign, not local accuracy, dominates global error.** The oracle-sign rows use GT and are not an inference system. Literature numbers (DCL-Net, TUS-REC2024) are on real data with different metrics and are not directly comparable.

**T1 rerun on the new backbone** (5000 steps; Q1 random L1 now matches the per-frame-norm architecture). Macro-CT in-FOV EPE (px):

| Gap | Identity | Q0 | Q1 | **Q2** | Q3 (composed L1 head) |
|---|---:|---:|---:|---:|---:|
| 1 | 1.676 | 1.671 | 1.682 | **0.953** | 0.511 |
| 2 | 3.100 | 3.089 | 3.090 | **1.226** | 1.020 |
| 4 | 5.886 | 5.845 | 5.826 | **2.265** | 1.924 |

Q2 beats Q0/Q1 on every CT and gap (old Q2: 1.344/2.194/4.520), with `|z'|` MAE 0.48/0.66/1.13 mm. Q2 is **still worse than Q3** at every gap: the point-query MLP does not extract all the information present in the frozen features. Q0 and Q1 remain ≈ identity.

**Relative elevation sign, Housden-style 3-frame test** (`triangle` variant in `scripts/eval_local_drift.py`). From L1 `|dz|` at gaps 1 and 2 (gap 2 = L1 run on stride-2 clips), each step is "same direction" if `|d(t,t+2)|` is closer to `a+b` than to `|a-b|`; signs are chained and one GT bit per sweep fixes the global mirror ambiguity. VAL, new L1, full sweeps:

| | Per-step reversal accuracy | Chained sign accuracy | GPE (mm) | FDE (% path) |
|---|---:|---:|---:|---:|
| Predicted `|dz|` | 92.6% | 65.9% | 6.27 | 45.0% |
| GT `|dz|` (ceiling of the test) | 99.7% | 93.5% | — | — |

Image-only relative sign reduces GPE from 9.64 (no sign) to 6.27 mm, against 2.95 mm with the full oracle sign. The local test works; the loss comes from chaining (one wrong step flips all later frames).
