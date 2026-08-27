"""NR7 suite driver: TACR double-Q (mean) + UNIFORM sampler + NR7 family.

The comparison arm is the other agent's fix_a_mean (double-Q mean +
family-BALANCED sampler, 31 policies, no nr7). This suite tests the user's
hypothesis: natural diversification (a 5th family, the NR7 breakout
signal) + the paper's original uniform-over-policies sampler stabilizes
the BC prior better than artificial family re-weighting.

5 seeds at the standard 3k budget (10 x 300), 2-parallel. Tag:
fix_a_mean_nr7 (checkpoints under src/models/tacr/checkpoints/tacr/).
"""
from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [20260814, 1, 2, 3, 4]
TAG = "fix_a_mean_nr7"

CMD = [
    "python", "-m", "src.models.tacr.train",
    "--use-double-q", "--double-q-mode", "mean",
    "--no-balanced-families",
    "--tag", TAG,
]


def run(seed: int) -> None:
    cmd = CMD + ["--seed", str(seed)]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(run, SEEDS))
    print(f"=== suite complete: {TAG} ===")