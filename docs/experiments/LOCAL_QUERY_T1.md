# CT2Eco Gate T1 — frozen Local Encoder, learned point query

**Goal:** Test whether an independently trained query readout extracts reusable geometric information from frozen L1 representations. T0 cosine retrieval failed; the pretrained L1 dense motion head remains a strong but **non-independent supervised reference**.

## Protocol

* Query `(u,v,t=0,t'=gap)`; `gap=1,2,4`. Sample a deterministic grid of 64 queries inside frame 0, **without using target poses to choose queries**.
* GT `target UV, |z|` is calculated with `inv(T_target) @ T_source` by `project_queries()` and is used **only for targets, loss and metrics**. Projected 2D points at large `|z|` may be invisible anatomically: this is a synthetic pose-projection task, **not** validated correspondence in real ultrasound.
* Q0 = no-image query-only baseline; Q1 = untrained random initialized L1 frozen, seed 314159; Q2 = pretrained L1 frozen from `mixed_dense_l1_s17/best.pt`. **Same readout, initialization, training schedule, in-FOV labels and validation queries.**
* Readout receives source/target frame features at the source query position; average local transition feature; mean global transition context; normalized query coordinates and gap. It regresses projected target `(u',v')` and nonnegative `|z'|`.
* Q3 = existing supervised L1 xyz head's **2D** displacement, composed over gap, for reference. Its elevation is *not* composed across gaps because signed elevation is unknown. It is neither a frozen-transfer baseline nor a new pretraining test.
* Frozen L1 weights never update. No NCC or L3 residual is required. No Global Stream or Transformer is introduced.
* All in-FOV projected queries are used for training and primary evaluation (not only near-zero `|z|`). Validation also reports `|z|` ranges 0–0.5, 0.5–1.5, 1.5–3, ≥3 mm. |z|≤0.5 mm is a **proxy** for overlap only.
* Train cases only for fitting, independent val cases for checkpoint selection. Checkpoints selected using mean per-CT UV EPE over gaps 1/2/4. **Never select or evaluate on TEST.**
* CT-to-CT sampling is unbalanced in the raw training data; interpret metrics per CT, and do not treat thousands of correlated queries as independent cases.
* This is supervised readout transfer within the same synthetic geometry family on the same train CT split; it is **not independent-domain or independent-task pretraining evidence**.

## Install & run

Apply this patch **after** `ct2eco-local-frozen-transfer-t0.patch`.

```bash
python -m pytest -q tests/test_local_query_t1.py tests/test_local_transfer.py

for arm in q0 q1 q2; do
  python scripts/train_local_query_t1.py \
    --data data/pilot_v2 \
    --l1-checkpoint runs/local/mixed_dense_l1_s17/best.pt \
    --variant "$arm" --steps 1500 --batch 4 --queries 64 \
    --val-clips-per-sweep 8 --eval-every 250 --seed 17 \
    --out "runs/local/t1_${arm}_s17"
done

python scripts/report_local_query_t1.py \
  --runs-root runs/local --seed 17 \
  --out runs/local/t1_transfer_confirmation.json
```

Each arm writes `history.jsonl`, `last.pt`, `best.pt`, `best_val.json`. For checkpoint-only validation, use `--eval-only --init runs/local/t1_q2_s17/best.pt` with the same other arguments, save to a new `--out` folder. JSON reports contain gap 1/2/4 micro- and macro-CT EPE in native pixels, |z| MAE in mm, recall within 4/8 px and per-CT results.

## Readout and decision gate

Primary: **Q2 EPE below both Q0 and Q1** on CT-macro mean, consistently across gaps and CT cases. Compare to Q3 supervised motion-head XY reference to understand the remaining gap, but Q3 cannot prove transfer. Secondary: elevational |z| MAE (not a signed z evaluation). If Q2 fails while Q3 performs well, move to a probe on **transition features** that explicitly matches the original motion feature training objective before changing the backbone. Never interpret failure of cosine similarity as proof of absence of geometric information.

**Known limitations:** 3 independent validation CTs; all query heads trained on task-adjacent labels; random-vs-pretrained features share L1 architecture; 2D pose projection is sometimes physically unobservable; no explicit acoustic beam or confidence model. Further evidence needs more held-out CTs and eventually novel tasks/datasets, not endless selection on the same 3 CTs.
