# Decisions, failed hypotheses and next gates

**As of 2026-10-17.** The numeric evidence is in [`RESULTS.md`](RESULTS.md).

## Confirmed research decisions

| Topic | Decision | Basis / limitation |
|---|---|---|
| Original US-D4RT | Retain as historical baseline; do not equate earlier loss decreases with image-dependent geometry | Sanity checks in `EXPERIMENTS.md` showed reliance on query priors |
| **Local backbone** | Preserve **L1 temporal 3-D CNN** as primary local encoder v1 | Mixed-motion `dx,dy,|dz|` better than spatial prior and L2b; T1 Q2 improves frozen transfer controls |
| L2b end-to-end NCC fusion | Do not promote to default | Improves pure movements but degrades combined 3-axis motion |
| NCC physical role | Keep as a scientifically informative optional diagnostic/branch | Oracle compensation and branch ablations show elevation signal, yet gains over simple features are small |
| Residual elevation | Keep **R0** as simple reference and **R2** as optional accuracy variant | R2 average +4.46% `|dz|` MAE improvement vs R0 across seeds, but CT×seed not uniformly positive; latency +15.9% in current implementation |
| T0 cosine descriptor matching | Treat as failed diagnostic, **not** evidence of no geometry in L1 features | Trained cosine matching loses to identity, while supervised head performs well |
| **T1 decoder on frozen L1** | Mark aggregate UV EPE gate **passed** | Q2 beats query-only and random-L1 controls on gaps 1/2/4; is same-family supervised query transfer |

## Current limitations

1. **Small independent validation set:** 3 CT; repeated pixels/queries do not provide independent samples.
2. **Rigid/static synthetic anatomy** and SonoGym acquisition assumptions; no proof yet of robustness to deformation, realistic probe pressure, distinct ultrasound scanners or real video.
3. **Signed elevation/complete 6-DoF observability** not established. A low error on `|dz|` does not imply recovery of sign.
4. **Point projections are geometry labels, not always visible image correspondences** across changes of image plane.
5. **No independent cross-task/frozen transfer or global reconstruction result:** T1 is a new head trained on a related synthetic geometry task, not a demonstrated multipurpose foundation model.
6. **No test-set evaluation in these local-stage reports.** Validation checkpoint selection must not be mistaken for final test evidence.
7. Archived stepwise guides may say a module is 'future' at their historical date; this file is the current decision log.

## Next experiment: Global context Gate G0 — *planned, not implemented*

Goal: test whether global temporal context improves long-range query point transfer beyond the already trained T1 readout, without changing the Local L1 backbone. Use the same CT-disjoint validation queries; do not reselect on test.

| Arm | Description | Intended control |
|---|---|---|
| G0 | T1 Q2 lightweight query readout | Current local baseline |
| G1 | Larger-capacity local query MLP | Controls for parameter count |
| G2 | Frozen L1 + small temporal context model + query readout | Tests whether global temporal information helps |
| G3 | G2 with shuffled/perturbed temporal order | Tests sensitivity to ordered context |

Measure UV EPE at gaps 1/2/4 (emphasis on long gaps), `|z'|` MAE, per-CT differences, in-plane visibility strata, and consistency of direct transfers versus composed transfers. The temporal model and its benefit are **not yet demonstrated**.

**Gate interpretation:** advance only if G2 outperforms capacity-matched G1 on held-out CT, without regressions in elevation and with meaningful order dependence. Follow with multiple seeds / larger independent CT confirmation and a locked test evaluation when architecture selection is over. Do not claim a full D4RT-like generalized foundation model from these local-stage results.

## Update 2026-10-18 — L1 v1.1

| Topic | Decision | Basis |
|---|---|---|
| **Local backbone** | Replace `mixed_dense_l1_s17` with **`l1v11_fnorm_var_s17`** (per-frame GroupNorm, clips 5–17, 15k steps) | Val MAE 0.308 → 0.160 mm on full sweeps; accuracy independent of clip length (RESULTS §6). One seed only; multi-seed confirmation deferred (not a priority) |
| NCC / L2b / L3 line | **Stop** | Gains ≈ 4% on 3 CT vs. ~2× from training/normalization fixes |
| T1 readout | Gate "Q2 ≥ Q3" **not passed** | Q2 0.95/1.23/2.27 px vs Q3 0.51/1.02/1.92 px; the readout, not the encoder, is the bottleneck |
| Global error | Elevation **sign** is the main obstacle | Image-only drift GPE 9.6 mm vs 11.6 static; 2.95 mm with oracle sign |

Implications for Gate G0: (1) include Q3 composition as a baseline; (2) the query readout needs spatial context (e.g. local feature patches/conv features rather than single-point samples) before global gains can be attributed to temporal context; (3) report drift (GPE/FDE) alongside EPE, and treat recovery of the dz sign as the key question for the Global Stream.
