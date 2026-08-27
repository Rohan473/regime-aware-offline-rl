"""De-confounding arm: double-Q MIN + uniform sampler + NR7 family.

fix_a (dq-min, uniform, 31 pol, no nr7) is the best TACR arm measured
(+0.072 mean margin, 4/5). My fix_a_mean_nr7 differs from it in TWO knobs
(nr7 added AND dq mean-vs-min mode). This arm isolates NR7's marginal
effect: identical to fix_a except the nr7 family is in the dataset.
"""
from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [20260814, 1, 2, 3, 4]
TAG = "fix_a_nr7"

CMD = [
    "python", "-m", "src.models.tacr.train",
    "--use-double-q", "--double-q-mode", "min",
    "--no-balanced-families",
    "--tag", TAG,
]

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(
            lambda s: (print(f"$ {' '.join(CMD + ['--seed', str(s)])}", flush=True),
                       subprocess.run(CMD + ["--seed", str(s)], cwd=ROOT, check=True)),
            SEEDS,
        ))
    print(f"=== suite complete: {TAG} ===")