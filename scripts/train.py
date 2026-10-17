"""Train US-D4RT v0.1 on the point-correspondence proxy (L_geo only).

    python scripts/train.py --data data/pilot_v0 --out runs/v0_1
    python scripts/train.py --data data/pilot_v0 --out runs/fit_1case --subset s0344   # overfit check

Evaluation uses the fixed queries inside the first --clip frames of each sweep, on two sets:
fit = the training sweeps themselves (can the model learn the task at all?), val = held-out CTs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct2eco.model import USD4RT, geo_loss, pixel_mm  # noqa: E402
from ct2eco.queries import QueryDataset, split_sweeps  # noqa: E402


def unpack(batch, device):
    q = {k: v.to(device, non_blocking=True) for k, v in batch["queries"].items()}
    uv = torch.stack([q["source_u"], q["source_v"]], -1).float()
    gt = torch.stack([q["target_x_mm"], q["target_y_mm"], q["target_z_mm"]], -1)
    return batch["frames"].to(device), batch["spacing_mm"].to(device), q, uv, gt


def source_mm(uv, spacing, width):
    xy = pixel_mm(uv, spacing[:, None], width)
    return torch.cat([xy, torch.zeros_like(xy[..., :1])], -1)


@torch.no_grad()
def blind_offsets(loader, clip, device):
    """Per frame gap on the training queries: mean (target - source) displacement and mean |z|.
    The best image-free predictors with and without the elevational sign."""
    acc = torch.zeros(2 * clip - 1, 5, device=device)
    for batch in loader:
        frames, spacing, q, uv, gt = unpack(batch, device)
        dt = (q["target_frame"] - q["source_frame"] + clip - 1).flatten()
        d = (gt - source_mm(uv, spacing, frames.shape[-1])).flatten(0, 1)
        acc.index_add_(0, dt, torch.cat([d, d[:, 2:].abs(), torch.ones_like(d[:, :1])], 1))
    return acc[:, :4] / acc[:, 4:].clamp(min=1)  # (2 clip - 1, 4): mean dx, dy, dz, |dz|


@torch.no_grad()
def evaluate(model, loader, device, prefix, blind, clip):
    """Mean 3D error (mm), overall / in-slice / out-of-slice, against two image-free baselines:
    identity ('probe did not move') and blind (mean training displacement for that frame gap)."""
    model.eval()
    keys = ("all", "in_slice", "out_slice", "z_abs", "identity", "identity_z", "blind", "blind_z", "all_flip", "z_mag",
            "sign_ok", "x", "y", "identity_x", "identity_y", "z_mag_dt1", "identity_z_dt1", "blind_z_mag")
    sums = {k: torch.zeros(2, device=device) for k in keys}
    for batch in loader:
        frames, spacing, q, uv, gt = unpack(batch, device)
        with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
            pred = model(frames, spacing, q["source_frame"], q["target_frame"], uv)
        err = (pred - gt).norm(dim=-1)
        # sign-free views: best global z flip per video (eval batch = 1 video), and | |z_pred| - |z_gt| |
        flipped = (pred * pred.new_tensor([1, 1, -1]) - gt).norm(dim=-1)
        err_flip = flipped if flipped.mean() < err.mean() else err
        z_mag = (pred[..., 2].abs() - gt[..., 2].abs()).abs()
        src = source_mm(uv, spacing, frames.shape[-1])
        b = blind[q["target_frame"] - q["source_frame"] + clip - 1]
        bl = src + b[..., :3]
        ins = q["target_in_slice"].bool()
        dt1 = (q["target_frame"] - q["source_frame"]).abs() == 1  # adjacent frames: where speckle is most informative
        for k, e, m in (("all", err, None), ("in_slice", err, ins), ("out_slice", err, ~ins),
                        ("z_abs", (pred[..., 2] - gt[..., 2]).abs(), None),
                        ("identity", (src - gt).norm(dim=-1), None), ("identity_z", gt[..., 2].abs(), None),
                        ("blind", (bl - gt).norm(dim=-1), None), ("blind_z", (bl[..., 2] - gt[..., 2]).abs(), None),
                        ("all_flip", err_flip, None), ("z_mag", z_mag, None),
                        ("sign_ok", (err.mean() <= flipped.mean()).float()[None], None),  # video got the direction right
                        ("x", (pred[..., 0] - gt[..., 0]).abs(), None), ("y", (pred[..., 1] - gt[..., 1]).abs(), None),
                        ("identity_x", (src[..., 0] - gt[..., 0]).abs(), None),
                        ("identity_y", (src[..., 1] - gt[..., 1]).abs(), None),
                        ("z_mag_dt1", z_mag, dt1), ("identity_z_dt1", gt[..., 2].abs(), dt1),
                        ("blind_z_mag", (b[..., 3] - gt[..., 2].abs()).abs(), None)):
            m = torch.ones_like(e, dtype=torch.bool) if m is None else m
            sums[k] += torch.stack([e[m].sum(), m.sum()])
    model.train()
    return {f"{prefix}_{k}" + ("" if k == "sign_ok" else "_mm"): (s[0] / s[1].clamp(min=1)).item() for k, s in sums.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--subset", nargs="+", help="train only on these sweep ids or case ids (e.g. s0344)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--clip", type=int, default=16, help="training frames per video")
    p.add_argument("--queries", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--init", type=Path, help="start from this checkpoint's weights (curriculum / fine-tuning)")
    p.add_argument("--frame-tokens", action="store_true", help="per-frame register token, query gets g[t_r] - g[t_s]")
    p.add_argument("--residual", action="store_true", help="predict the displacement from the source point")
    p.add_argument("--beta-mm", type=float, default=1.0, help="SmoothL1 beta in mm (~0 = L1, as D4RT)")
    p.add_argument("--local-patch", type=int, default=0, help="k: raw k x k source pixels per query (D4RT uses 9)")
    p.add_argument("--early-fusion", action="store_true", help="encoder sees [t-1, t, t+1] stacked per frame")
    p.add_argument("--sign-invariant", action="store_true", help="geo loss up to a per-video flip of z (diagnostic)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    names = [d.name for d in split_sweeps(a.data, "train")]
    subset = None if not a.subset else [n for n in names if any(n == t or n.startswith(t + "_") for t in a.subset)]
    train_ds = QueryDataset(a.data, "train", a.queries, clip_len=a.clip, sweeps=subset)
    # sampling with replacement: a one-sweep subset still fills every batch
    sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=a.batch * a.eval_every)
    train = torch.utils.data.DataLoader(train_ds, a.batch, sampler=sampler, num_workers=a.workers,
                                        drop_last=True, persistent_workers=a.workers > 0)
    evals = {name: torch.utils.data.DataLoader(QueryDataset(a.data, split, fixed=True, clip_len=a.clip, sweeps=sw), 1,
                                               num_workers=a.workers)
             for name, split, sw in (("fit", "train", subset), ("val", "val", None))}

    blind = blind_offsets(evals["fit"], a.clip, a.device)
    model = USD4RT(early_fusion=a.early_fusion, local_patch=a.local_patch, residual=a.residual,
                   frame_tokens=a.frame_tokens).to(a.device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=a.device)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    rounds = a.steps // a.eval_every
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=rounds * len(train),
                                                pct_start=max(0.05, 2 / (rounds * len(train))))
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, {len(train_ds)} train sweeps", flush=True)

    best, log = float("inf"), open(a.out / "log.jsonl", "a")
    for rnd in range(rounds):
        t0, total = time.time(), 0.0
        for batch in train:
            frames, spacing, q, uv, gt = unpack(batch, a.device)
            with torch.autocast("cuda", torch.bfloat16, enabled=a.device == "cuda"):
                pred = model(frames, spacing, q["source_frame"], q["target_frame"], uv)
            loss = geo_loss(pred, gt, a.sign_invariant, a.beta_mm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        stats = {"step": (rnd + 1) * len(train), "train_loss": total / len(train), "time_s": time.time() - t0}
        for name, loader in evals.items():
            stats |= evaluate(model, loader, a.device, name, blind, a.clip)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}), flush=True)
        log.write(json.dumps(stats) + "\n")
        log.flush()
        ckpt = {"model": model.state_dict(), "step": stats["step"],
                "args": vars(a) | {"data": str(a.data), "out": str(a.out), "init": str(a.init)}}
        torch.save(ckpt, a.out / "last.pt")
        if stats["val_all_mm"] < best:
            best = stats["val_all_mm"]
            torch.save(ckpt, a.out / "best.pt")


if __name__ == "__main__":
    main()
