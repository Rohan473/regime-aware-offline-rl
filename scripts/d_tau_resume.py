"""Resume the τ sweep: run only the 28 jobs that did not complete.

Run: python scripts/d_tau_resume.py
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEEDS = [20260814, 1, 2, 3, 4]
VARIANTS = ["d_minus_fuzzy", "d"]
SWEEPS = [("--tau", [0.001, 0.0025, 0.005, 0.01, 0.02], "d_tau_polyak", "polyak"),
          ("--expectile", [0.5, 0.6, 0.7, 0.8, 0.9], "d_tau_exp", "expectile")]
BASE = ROOT / "src/models/d/checkpoints"


def is_done(tag_root: str, v: float, var: str, s: int) -> bool:
    d = BASE / tag_root / f"{v:g}" / var / f"s{s}"
    return (d / "d_final.pt").exists() and (d / "d_best.pt").exists()


def main() -> None:
    jobs = []
    for flag, vals, tag_root, label in SWEEPS:
        for v in vals:
            for var in VARIANTS:
                for s in SEEDS:
                    if not is_done(tag_root, v, var, s):
                        jobs.append((flag, v, var, s, tag_root, label))

    print(f"Resuming {len(jobs)} missing jobs...", flush=True)

    def run(job):
        flag, v, var, s, tag_root, label = job
        cmd = ["python", "-m", "src.models.d.train", "--variant", var,
               "--seed", str(s), "--tag", f"{tag_root}/{v:g}", flag, str(v)]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            print(f"!!! {var} s{s} {label}={v:g} rc={r.returncode}", flush=True)
            print((r.stderr or "")[-800:], flush=True)
            return
        last = [l for l in (r.stdout or "").strip().splitlines() if l][-1]
        print(f"ok {var} s{s} {label}={v:g}: {last}", flush=True)

    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(run, jobs))
    print("ALL TAU RESUME TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()