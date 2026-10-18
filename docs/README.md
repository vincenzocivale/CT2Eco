# CT2Eco — Research documentation

**Snapshot:** 2026-10-18 (L1 v1.1 update, see RESULTS §6). **Current status:** Local Stream L1 trained; L3 residual experiments and frozen-feature transfer Gates T0/T1 evaluated on validation. Global Stream and final D4RT-style decoder remain **future work**.

This folder distinguishes the experimental *current state* from the chronological protocols. The original US-D4RT baseline and failed experiments remain in [`../EXPERIMENTS.md`](../EXPERIMENTS.md); they must not be conflated with the later Local Stream.

| Start here | Content |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Data geometry, implemented models, outputs, leakage boundaries |
| [`RESULTS.md`](RESULTS.md) | Consolidated numerical validation findings, baselines and comparability caveats |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Dataset layout, commands, checkpoints and verification |
| [`DECISIONS.md`](DECISIONS.md) | Architecture decisions, failed hypotheses, open gates and roadmap |
| [`experiments/`](experiments/) | Original step-by-step Local Stream protocols, preserved as historical references |

## Evidence rules

1. Numeric results below are taken from experiment logs shared up to 2026-10-17, not independently rerun on the research workstation in this documentation-only change.
2. Experimental selection uses **CT-disjoint train/validation**; **test has not been opened** in the local-stream experiments described here.
3. Thousands of pixels/queries are correlated within only **three validation CT cases**. The number of pixels is not the independent sample size.
4. Unless explicitly indicated, a model's “features” are **not** proven to represent persistent anatomical point identities. Pose-based projections can leave the image plane.
5. Probe results are not results for an independently validated ultrasound foundation model.

## Detailed protocol archive

| Original protocol | Main scope |
|---|---|
| [`experiments/LOCAL_STREAM.md`](experiments/LOCAL_STREAM.md) | L0/L1/L2 feasibility, isolated motions and pose diagnostics |
| [`experiments/LOCAL_NCC.md`](experiments/LOCAL_NCC.md) | NCC baseline and L2b dense probe |
| [`experiments/LOCAL_NEXT.md`](experiments/LOCAL_NEXT.md) | Pure-motion comparison, NCC ablation, combined 6-DoF |
| [`experiments/LOCAL_ALIGNMENT.md`](experiments/LOCAL_ALIGNMENT.md) | Fixed / oracle / estimated NCC alignment |
| [`experiments/LOCAL_L1_ALIGNMENT.md`](experiments/LOCAL_L1_ALIGNMENT.md) | Frozen L1 alignment versus classical matching |
| [`experiments/LOCAL_L3.md`](experiments/LOCAL_L3.md) | R0/R1/R2 residual elevation architecture |
| [`experiments/LOCAL_L3_CONFIRMATION.md`](experiments/LOCAL_L3_CONFIRMATION.md) | Three seeds and inference cost |
| [`experiments/LOCAL_FROZEN_TRANSFER.md`](experiments/LOCAL_FROZEN_TRANSFER.md) | Gate T0 zero-training point retrieval |
| [`experiments/LOCAL_QUERY_T1.md`](experiments/LOCAL_QUERY_T1.md) | Gate T1 trained decoder with frozen L1 |

The archived guides describe **what to run**; `RESULTS.md` records **what was actually observed**. Some archive notes were written before later experiments and contain prospective decisions superseded by `DECISIONS.md`.
