"""Objective Lab — idea 16 of next_experiments.txt.

"Does RL learn a trading representation or a reward representation?"

Four models share ONE encoder (a 128-d single-layer GRU over the shared
20x8 z-feature window, decision-date grid, split dates, and seed). Only the
objective head differs:

  A  predictive      MSE against the next-day market return (standardized)
  B  DSR             differential-Sharpe-ratio policy (DDR's reward)
  C  actor-critic    A2C policy + bootstrapped value (TACR's objective type)
  D  self-supervised masked reconstruction of the CURRENT-day features from
                     prior days in the window

By construction architecture/capacity/data/seed are identical across arms,
so any difference in the learned h_t (probes, effective rank, CKA) is
attributable to the objective. h_t is the final hidden state of the shared
encoder — the representation each head consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ARMS = ("A_predictive", "B_dsr", "C_actorcritic", "D_masked")
ROOT_CKPT = Path(__file__).resolve().parent / "checkpoints"


@dataclass
class ObjectiveLabConfig:
    window: int = 20
    hidden: int = 128
    in_dim: int = 8
    lr: float = 1e-3
    epochs: int = 30
    seed: int = 20260814
    block: int = 20

    # arm B (DSR)
    eta: float = 0.01
    warmup_steps: int = 20
    vol_targeting: bool = True
    target_vol: float = 0.15
    vol_target_window: int = 20
    max_leverage: float = 2.0
    cost_bps: float = 0.0

    # arm C (actor-critic)
    gamma: float = 0.99
    ac_sigma: float = 0.15
    ent_coef: float = 0.003

    # arm D (masked reconstruction)
    mask_ratio: float = 0.5

    checkpoint_dir: Path = ROOT_CKPT


def tag_path(cfg: ObjectiveLabConfig, arm: str) -> Path:
    return cfg.checkpoint_dir / arm / f"s{cfg.seed}"