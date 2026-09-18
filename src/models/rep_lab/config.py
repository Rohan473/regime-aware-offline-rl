"""Representation Lab — Stage 1 of the rep x policy research design.

Three representation learners share the SAME encoder (the 128-d GRU window
of the Objective Lab) and differ only in the unsupervised/supervised
objective that shapes h_t:

  auto         X -> encoder -> h -> decoder -> X          (reconstruction)
  predictive   h -> (R_{t+1}, R_{t+5}, |R_{t+1}|, sig_{t+5})  (multi-task)
  contrastive  two augmented views of the window -> InfoNCE (self-supervised)

The encoders are FROZEN after training; the probe battery and the downstream
policies evaluate the learned h_t. This makes the learned market-state
representation the explicit research object instead of an invisible step
inside end-to-end RL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

OBJECTIVES = ("auto", "predictive", "contrastive")
ROOT_CKPT = Path(__file__).resolve().parent / "checkpoints"

PRED_TARGETS = ("r1", "r5", "abs1", "vol5")


def _canonical_cols() -> tuple[str, ...]:
    from src.models.ddr.data import Z_COLUMNS

    return tuple(Z_COLUMNS)


@dataclass
class RepLabConfig:
    window: int = 20
    hidden: int = 128
    in_dim: int = 8
    lr: float = 1e-3
    epochs: int = 30
    seed: int = 20260814
    batch: int = 256

    # predictive: balanced MSE over the standardized future targets
    target_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)

    # contrastive augmentations (all applied in z-feature space)
    tau: float = 0.2          # InfoNCE temperature
    proj_dim: int = 128
    fea_mask_prob: float = 0.15   # cell-mask probability per view
    time_mask_prob: float = 0.15  # whole-row mask probability per view
    noise_std: float = 0.05       # additive z-noise std per view

    # downstream RL (frozen encoder, same A2C head as Objective Lab arm C)
    gamma: float = 0.99
    ac_sigma: float = 0.15
    ent_coef: float = 0.003
    cost_bps: float = 0.0
    cql_bc_coef: float = 0.5  # TD3+BC-style actor regularizer for CQL(H)

    # feature-scaling / latent-dimension sweeps: the causal feature subset fed
    # to the encoder (None -> canonical 8) and a variant tag that keeps sweep
    # checkpoints separate from the headline ones.
    feature_cols: tuple[str, ...] | None = None
    tag: str = ""

    # cross-market: process a different market's processed dir (None -> SPY)
    processed_dir: Path | None = None

    checkpoint_dir: Path = ROOT_CKPT

    def feature_columns(self) -> tuple[str, ...]:
        return tuple(self.feature_cols) if self.feature_cols else _canonical_cols()

    def n_features(self) -> int:
        return len(self.feature_columns())


def tag_path(cfg: RepLabConfig, objective: str, subdir: str = "reps") -> Path:
    name = objective if not cfg.tag else f"{objective}_{cfg.tag}"
    return cfg.checkpoint_dir / subdir / name / f"s{cfg.seed}"