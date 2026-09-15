"""β sweep on the 32-policy unweighted DNF dataset.

Goal: does lowering β (sharper AWR concentration) stabilize the 32p
model by keeping the actor long-dominant — or does it collapse Sharpe by
over-concentrating on a narrow advantage slice?

Pre-registered success criteria:
  (1) Does any β achieve final-epoch PASS on 32p-unweighted?
  (2) If so, does the corresponding β on comp-matched improve Sharpe > 0.837?
  (3) If no β stabilizes 32p, β is not the binding constraint.

Run: python scripts/d_beta_sweep.py
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
GRID = [0.5, 1.0, 1.5, 2.0, 3.0]
TAG_ROOT = "d_beta"


def main() -> None:
    jobs = []
    for beta in GRID:
        for variant in VARIANTS:
            for seed in SEEDS:
                jobs.append((beta, variant, seed))

    def run(job):
        beta, variant, seed = job
        cmd = ["python", "-m", "src.models.d.train", "--variant", variant,
               "--seed", str(seed), "--tag", f"{TAG_ROOT}/{beta:g}",
               "--temperature", str(beta)]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            print(f"!!! {variant} s{seed} b={beta} rc={r.returncode}", flush=True)
            print((r.stderr or "")[-800:], flush=True)
            return
        last = [l for l in (r.stdout or "").strip().splitlines() if l][-1]
        print(f"ok {variant} s{seed} b={beta}: {last}", flush=True)

    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(run, jobs))
    print("ALL BETA SWEEP TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()