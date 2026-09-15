"""Train all four Objective Lab arms (idea 16) on the shared encoder.

Usage:
    python scripts/objective_lab.py [--epochs N] [--hidden N] [--seed N] [A B C D]

Trains A_predictive, B_dsr, C_actorcritic, D_masked on the identical DDR
grid/window/splits with the same seed, then prints their val metric and
checkpoint paths. Checkpoints: src/models/objective_lab/checkpoints/<arm>/s<seed>/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.objective_lab.config import ARMS, ObjectiveLabConfig, tag_path
from src.models.objective_lab.train import train_arm


def main() -> None:
    args = sys.argv[1:]
    only = [a for a in args if "_" not in a and a in ARMS]
    cfg = ObjectiveLabConfig()
    if "--epochs" in args:
        cfg.epochs = int(args[args.index("--epochs") + 1])
    if "--hidden" in args:
        cfg.hidden = int(args[args.index("--hidden") + 1])
    if "--seed" in args:
        cfg.seed = int(args[args.index("--seed") + 1])
    arms = only or list(ARMS)

    print(f"Objective Lab: arms={arms}, seed={cfg.seed}, epochs={cfg.epochs}, hidden={cfg.hidden}")
    for arm in arms:
        t0 = time.time()
        agent, hist, ckpt = train_arm(arm, cfg)
        print(f"[{arm}] best val_metric {hist['val_metric'].max():.4f}  "
              f"({time.time() - t0:.0f}s) -> {ckpt}")


if __name__ == "__main__":
    main()