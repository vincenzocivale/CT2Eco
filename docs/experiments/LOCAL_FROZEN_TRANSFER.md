# CT2Eco: Gate T0 — Zero-shot sparse correspondence with frozen L1

**Question**: Do frozen L1 `frame_dense` descriptors retain enough local information to
recover point correspondences across 1, 2 or 4 frames, *without training a
query head*? This is a diagnostic, NOT a claim of complete geometric
foundation-model transfer.

The trained backbone is the existing mixed-motion L1 checkpoint. It is never
updated. The geometry supervision used to train its original dense motion head
means that re-evaluating that head **does not qualify as a new task**. It is
retained only as a strong geometric baseline alongside independent descriptor
matching.

## Four important cautions

1. CT2Eco synthetic pose projections do not necessarily describe *visible*
   B-mode correspondence after out-of-plane motion. Report both the entire
   in-FOV set and strata in |dz|. `proxy_overlap` = |dz|<=0.5 mm (an
   exploratory *proxy*, not a guarantee of actual acoustic overlap).
2. L1's `frame_dense` has spatial stride four; zero-shot nearest-neighbor
   retrieval is quantized to four raw pixels. The trained L1 head gives
   subpixel interpolation, so compare against random-init same-backbone
   retrieval and raw-patch NN, not only against that trained head.
3. A failed cosine-NN experiment does not prove the representations are
   uninformative. The subsequent Gate T1, *if warranted*, trains a small
   query-conditioned readout while freezing L1 and compares to random-init
   frozen L1 and coordinates-only heads.
4. Independent validation units are CT cases, not queried pixels. Use per-case
   results and do not unlock test during method selection.

## Methods (same queries / pose-derived labels)

* `identity`: query stays at same image pixel.
* `raw_patch`: normalized B-mode patches, NCC-like cosine NN, local 2D search.
* `random_l1`: `frame_dense` cosine-NN of an untrained L1 architecture.
* `trained_l1`: cosine-NN of frozen pretrained L1 `frame_dense`.
* `trained_motion_head`: preexisting L1 dense xyz head; adjacent motion vectors
  are composed for gaps 2/4. This head was supervised on dense geometry!

All nearest-neighbor searches use **fixed query-centered local windows**, not
windows centered on ground-truth targets. Ground-truth poses enter only
`project_queries` after the image-only predictions have been produced.

## Run

From the repository root, after applying the patch:

```bash
python scripts/eval_local_frozen_transfer.py \
  --data data/pilot_v2 \
  --l1-checkpoint runs/local/mixed_dense_l1_s17/best.pt \
  --out runs/local/frozen_transfer_t0 \
  --clips-per-sweep 8 --batch 4 \
  --queries 64 --radius 8 --raw-patch 9 \
  --proxy-max-dz-mm 0.5
```

Results: `runs/local/frozen_transfer_t0/frozen_transfer_val.json`. For each
gap, inspect `micro.proxy_overlap`, `micro.fov_all`, the `z_*` bins,
`macro_ct_epe_px` and `by_ct`. Check `search_reachable_fraction` to
measure how many true points lie within the prescribed local search window;
**do not filter out** points that do not. Each modality reports EPE in **native image
pixels** and recall within 4/8 pixels. Do not compare the value to prior
L3 elevation MAE in millimetres: different task, support and units.

### Decision to advance to Gate T1

* If trained frozen feature matching outperforms random-init and raw-patch
  matching on most CTs in high-overlap bins, train a *small point-query head*
  while keeping L1 frozen. Show performance also at gaps 2/4.
* If feature NN matching fails but the trained motion head is good, investigate
  whether `frame_dense` is not descriptor-aligned (it was trained for motion,
  not contrastive matching). A small frozen feature query head may still work;
  this would be a separate supervised transfer experiment.
* If neither NN nor motion-head composition beats identity, audit geometry,
  data overlap, and measurement first; do not start Global Stream yet.

## Tests

```bash
python -m pytest -q tests/test_local_transfer.py
```
