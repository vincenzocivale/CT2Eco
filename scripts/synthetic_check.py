"""Synthetic gates for the model: can it read the video at all? Each runs in ~1 minute on a GPU.

    python scripts/synthetic_check.py --tasks image product shift [--model '{"early_fusion": true}'] [--fp32]

image    x = 5 c                 c = width of a bright band (target depends on the video only)
product  x = c * (t_r - t_s)     same cue, multiplied by the frame gap (the structure of our real task)
shift    x = v * (t_r - t_s)     random texture translating laterally by v px/frame, v in [-3, 3]: motion must be
                                 measured between frames, as in the lateral sanity dataset
Pass = final error < 30% of the 'predict 0' error, and no collapse after the best evaluation.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ct2eco.model import USD4RT, geo_loss  # noqa: E402

H, W, SPACING = 200, 150, 0.4


def batch(task, B, T=8, N=128, device="cuda"):
    c = torch.rand(B, device=device) * 4 - 2
    if task == "shift":
        v = c * 1.5  # px / frame
        big = torch.rand(B, 1, H, W + 64, device=device)
        big = F.avg_pool2d(big, 3, 1, 1)  # speckle-like grain of a few pixels
        x0 = torch.arange(T, device=device)[None] * v[:, None] + 32  # (B, T) left edge of the window
        cols = (x0[..., None] + torch.arange(W, device=device)).round().long().clamp(0, W + 63)  # (B, T, W)
        frames = torch.stack([big[b, 0][:, cols[b]].permute(1, 0, 2) for b in range(B)])[:, :, None]
        speed_mm = -v * SPACING  # content moving right = probe moving left: target x decreases
    else:
        frames = torch.rand(B, T, 1, H, W, device=device) * 0.3
        width = ((c + 2) / 4 * (W - 20) + 10).long()
        frames = frames + 0.6 * (torch.arange(W, device=device)[None] < width[:, None]).float()[:, None, None, None, :]
        speed_mm = c
    src, ref = torch.randint(T, (B, N), device=device), torch.randint(T, (B, N), device=device)
    uv = torch.rand(B, N, 2, device=device) * torch.tensor([W - 1, H - 1], device=device)
    gt = torch.zeros(B, N, 3, device=device)
    gt[..., 0] = 5 * c[:, None].expand(-1, N) if task == "image" else speed_mm[:, None] * (ref - src)
    if task != "image":  # make the in-plane target consistent with the source point, as in the real labels
        gt[..., 0] += (uv[..., 0] - (W - 1) / 2) * SPACING
        gt[..., 1] = uv[..., 1] * SPACING
    return frames.clamp(0, 1), torch.full((B, 2), SPACING, device=device), src, ref, uv, gt


def run(task, model_kw, steps, lr, warmup, fp32, device="cuda", seed=0):
    torch.manual_seed(seed)
    model = USD4RT(**model_kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(warmup, 1)))
    curve = []
    for step in range(1, steps + 1):
        f, s, src, ref, uv, gt = batch(task, 4, device=device)
        with torch.autocast("cuda", torch.bfloat16, enabled=not fp32):
            loss = geo_loss(model(f, s, src, ref, uv), gt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % (steps // 8) == 0:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=not fp32):
                f, s, src, ref, uv, gt = batch(task, 32, device=device)
                pred = model(f, s, src, ref, uv).float()
                src_x = (uv[..., 0] - (W - 1) / 2) * SPACING if task != "image" else 0
                err = (pred[..., 0] - gt[..., 0]).abs().mean().item()
                zero = (gt[..., 0] - src_x).abs().mean().item()
            model.train()
            curve.append(round(err / zero, 2))
    ok = curve[-1] < 0.3 and curve[-1] <= min(curve) + 0.1
    return ok, curve


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=["image", "product", "shift"])
    p.add_argument("--model", default="{}", help="JSON kwargs for USD4RT")
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--fp32", action="store_true")
    a = p.parse_args()
    kw = json.loads(a.model)
    for task in a.tasks:
        ok, curve = run(task, kw, a.steps, a.lr, a.warmup, a.fp32)
        print(f"{'PASS' if ok else 'FAIL'} {task:8s} {kw} lr={a.lr} warmup={a.warmup} fp32={a.fp32}  error/zero: {curve}", flush=True)


if __name__ == "__main__":
    main()
