"""Representation learners + downstream heads for the Representation Lab.

``encoder`` is shared across all three objectives (imported from the
Objective Lab so the footprint is identical); only ``head`` differs.
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.objective_lab.model import Encoder


class FullReconHead(nn.Module):
    """Decoder baseline: reconstruct the whole flattened window from h_t."""

    def __init__(self, hidden: int, window: int, in_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(hidden, window * in_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


class PredictiveHeadM(nn.Module):
    """Multi-task head -> one scalar per standardized future target."""

    def __init__(self, hidden: int, n_targets: int) -> None:
        super().__init__()
        self.head = nn.Linear(hidden, n_targets)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


class Projector(nn.Module):
    """Nonlinear projector for InfoNCE (SimCLR-style 2-layer MLP)."""

    def __init__(self, hidden: int, proj_dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


class RawFeat(nn.Module):
    """Feature MLP mapping the flat raw window (window*in_dim) to hidden."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class DownstreamAgent(nn.Module):
    """A2C head consuming a FROZEN per-date feature vector.

    ``feat`` maps the raw input tensor to (B, hidden); for learned
    representations hidden is already the encoder output so feat is identity.
    """

    def __init__(self, hidden: int, feat: nn.Module | None = None) -> None:
        super().__init__()
        self.feat = feat if feat is not None else nn.Identity()
        self.policy = nn.Linear(hidden, 1)
        self.value = nn.Linear(hidden, 1)

    def mu(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.policy(self.feat(z)).squeeze(-1))

    def mu_raw(self, z: torch.Tensor) -> torch.Tensor:
        return self.policy(self.feat(z)).squeeze(-1)

    def v(self, z: torch.Tensor) -> torch.Tensor:
        return self.value(self.feat(z)).squeeze(-1)


def build_rep_objective(objective: str, cfg) -> nn.Module:
    from .config import OBJECTIVES

    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")

    class Rep(nn.Module):
        pass

    mod = Rep()
    mod.encoder = Encoder(cfg.in_dim, cfg.hidden)
    if objective == "auto":
        mod.head = FullReconHead(cfg.hidden, cfg.window, cfg.in_dim)
    elif objective == "predictive":
        from .config import PRED_TARGETS

        mod.head = PredictiveHeadM(cfg.hidden, len(PRED_TARGETS))
    else:  # contrastive
        mod.head = Projector(cfg.hidden, cfg.proj_dim)
    return mod