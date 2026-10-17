"""Compare runs on the last logged step (val set by default).

    python scripts/summarize_runs.py runs/v1_signinv runs/e1_fusion_signinv [--split fit]
"""

import argparse
import json
from pathlib import Path

COLS = [("all", "all"), ("all_flip", "all*"), ("z_abs", "z"), ("z_mag", "|z|"), ("z_mag_dt1", "|z|dt1"),
        ("x", "x"), ("y", "y"), ("sign_ok", "sign")]
BASE = [("identity", "all"), ("identity_z", "|z|"), ("blind_z_mag", "|z| blind"), ("identity_z_dt1", "|z|dt1"), ("identity_x", "x"), ("identity_y", "y")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--split", default="val", choices=["val", "fit"])
    a = p.parse_args()
    print(f"{'run':28s}" + "".join(f"{h:>8s}" for _, h in COLS) + "   (* = best z sign per video)")
    last = None
    for run in a.runs:
        last = [json.loads(line) for line in open(run / "log.jsonl")][-1]
        get = lambda k: last.get(f"{a.split}_{k}" + ("" if k == "sign_ok" else "_mm"))
        print(f"{run.name:28s}" + "".join(f"{get(k):8.2f}" if get(k) is not None else f"{'-':>8s}" for k, _ in COLS))
    print("baseline 'probe did not move': "
          + ", ".join(f"{h} {last[f'{a.split}_{k}_mm']:.2f}" for k, h in BASE if f"{a.split}_{k}_mm" in last))


if __name__ == "__main__":
    main()
