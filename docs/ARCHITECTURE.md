# Architecture and geometry

## Research question

Can synthetic CT-derived ultrasound sequences provide supervision for a visual encoder that learns usable local geometric representations, then supports query-based 3-D point transfer across multiple frames? The project is inspired by D4RT's query formulation, but the current Local Stream is not a trained foundation model and **does not yet include a Global Stream**.

## Synthetic data and coordinate system

- Prepared anatomical source: CT + TotalSegmentator-derived structures. SonoGym-derived ultrasound renderer, static synthetic anatomy and pose-controlled sweeps. Different sweeps from the **same CT stay in the same split**.
- Probe frame: `x` = image columns/lateral; `y` = rows/depth; `z` = elevation, with the image plane at `z=0`. Spatial quantities are in **millimetres**.
- Pixel `(u,v)` at image width `W` and spacing `(dy,dx)` maps to `((u-(W-1)/2)*dx, v*dy, 0)`.
- `T_t = poses[t]` maps probe coordinates to world/CT RAS. Source-to-target transformation is `inv(T_target) @ T_source`; its in-plane projection gives `(u',v')`, and the absolute elevational offset gives `|z'|`.
- Ground-truth poses and anatomy may generate **targets, validity masks, and oracle diagnostics**. They must **not** be forwarded to the image encoder or decoder as model inputs.
- Projection of a fixed 3-D point to a new probe plane is not automatically an acoustically visible point correspondence, particularly at large `|z'|`.

## Two distinct modeling phases

**Original US-D4RT v0.1** (`ct2eco/model.py`, original `README.md`, `EXPERIMENTS.md`) combines a CNN encoder, multi-view Transformer and cross-attention query decoder. Sanity checks showed that early runs relied heavily on blind temporal/position priors. Their numeric results are historical, not the baseline for Local L1 probes.

**Local Stream phase** (`ct2eco/local_stream.py`, `local_data.py`, `local_geometry.py`) explores image-dependent motion representation before the final global/query architecture:

| Variant | Implemented role | Status |
|---|---|---|
| L0 | Pairwise early-fused 2-D CNN | Feasibility control |
| **L1** | Temporal 3-D residual CNN, stride 1 in time; dense frame and transition maps (~1/4 spatial resolution, 128 channels). v1.1: `frame_norm=True` (GroupNorm per time step), trained on clips of 5–17 frames | **Frozen backbone** (`l1v11_fnorm_var_s17`) |
| L2 | L1 + feature correlation | Ablation, not primary |
| L2b | L1 + spatial B-mode NCC fusion | Useful on isolated motions; worse than L1 on combined motion |

`LocalStream` exposes `frame_dense` and `dense` transition features. `DenseDzHead`/the `xyz` head are *diagnostic supervised readouts*, not the future D4RT decoder. Training dense geometry from `inv(T_target) @ T_source` does not imply descriptor invariance, and `frame_dense` should not be assumed comparable by cosine similarity across frames.

## The L3 residual branch is a readout, not an updated encoder

Frozen L1 plus a residual elevated-distance head was evaluated in three matched arms:

- **R0:** frozen L1 visual features, no NCC input to the residual head.
- **R1:** same model capacity, plus fixed-coordinate NCC.
- **R2:** same capacity, plus fixed NCC **and** motion-compensated NCC warped with the **predicted** L1 in-plane flow.

L1 remains frozen and the head refines **only** `|dz|`: `dx,dy` are unchanged. For R2, the predicted mm displacement is converted to pixel coordinates using the sweep's spacing. R2 does not use pose-based oracle flow at inference. The oracle warp is diagnostic only in a separate experiment.

## Queries and transfer

- **T0:** frozen `frame_dense` features queried via zero-training cosine matching; compared with identity, raw B-mode patch matching, random L1 and the previously supervised motion head.
- **T1:** separate trainable lightweight point-query MLP, with **frozen** encoder. `Q0` uses coordinates/time only; `Q1` uses randomly initialized frozen L1; `Q2` uses pretrained frozen L1. `Q3` is the original supervised L1 head, not an independent pretraining control. Query target: projected `(u',v')` and `|z'|` at gaps 1, 2, 4.

The T1 readout works on local source/target features and transition context. It is **not** a global temporal Transformer; no cross-attention Global Stream has yet been trained.

## Observability boundaries

The magnitude of elevational motion can be reflected in speckle decorrelation even when the signed direction is ambiguous. Rotations and in-plane shifts both change fixed-position NCC. Matching/warping mitigates but does not eliminate this confound; oracle registration is not a universal performance ceiling when structures physically leave the plane. Do not claim full signed 6-DoF recovery or stable anatomical correspondence from the current results.
