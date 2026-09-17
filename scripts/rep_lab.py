"""Train the Representation Lab Stage 1 learners (and, optionally, the
downstream RL heads on frozen representations).

usage:
  python scripts/rep_lab.py [--objective {auto,predictive,contrastive,all}]
                            [--epochs N] [--downstream all]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.rep_lab.config import OBJECTIVES, RepLabConfig
from src.models.rep_lab.train import train_downstream, train_representation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="all", choices=list(OBJECTIVES) + ["all"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--downstream", default="none",
                    choices=["none", "all"])
    ap.add_argument("--seed", type=int, default=None,
                    help="override the training seed (extra seeds feed the "
                         "cross-seed stability axis of the quality scorecard)")
    args = ap.parse_args()

    cfg = RepLabConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.seed is not None:
        cfg.seed = args.seed

    objectives = list(OBJECTIVES) if args.objective == "all" else [args.objective]
    for obj in objectives:
        print(f"--- training representation objective: {obj} ---")
        _, _, ck = train_representation(obj, cfg)
        print(f"best -> {ck}")

    if args.downstream == "all":
        reps = ["raw"] + list(OBJECTIVES)
        for name in reps:
            print(f"--- training downstream A2C on frozen representation: {name} ---")
            _, _, ck = train_downstream(name, cfg)
            print(f"best -> {ck}")


if __name__ == "__main__":
    main()