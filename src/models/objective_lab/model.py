"""Shared encoder + objective heads for the Objective Lab.

``Encoder`` is the ONLY architecture the four arms share. Its state after
the window (h_t) is the representation under study. Heads are thin (they
only exist to create the objective's learning signal); the head weights
differ by design but the encoder is identical in footprint.
"""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    """Single-layer GRU over the (B, W, F) z-feature window -> h_t (B, D)."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, batch_first=True)

    def encode_all(self, x: torch.Tensor) -> torch.Tensor:
        """Full recurrent outputs (B, W, D)."""
        return self.rnn(x)[0]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Last-position hidden state h_t (B, D)."""
        return self.encode_all(x)[:, -1, :]


class PredictiveHead(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.head = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h).squeeze(-1)


class PolicyHead(nn.Module):
    """tanh-continuous action in [-1, 1] (arm B and C share the same form)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.head = nn.Linear(hidden, 1)

    def mu(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(h).squeeze(-1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mu(h)


class ValueHead(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.head = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h).squeeze(-1)


class ReconHead(nn.Module):
    """Predict the 8 z-features of the current day from h_t."""

    def __init__(self, hidden: int, in_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(hidden, in_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


def build_agent(arm: str, cfg) -> nn.Module:
    from .config import ARMS

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")

    class Agent(nn.Module):
        """(encoder, head) bundle; encoder weights are the shared skeleton."""

    mod = Agent()
    mod.encoder = Encoder(cfg.in_dim, cfg.hidden)
    if arm == "A_predictive":
        mod.head = PredictiveHead(cfg.hidden)
    elif arm in ("B_dsr", "C_actorcritic"):
        mod.policy = PolicyHead(cfg.hidden)
        if arm == "C_actorcritic":
            mod.value = ValueHead(cfg.hidden)
    elif arm == "D_masked":
        mod.head = ReconHead(cfg.hidden, cfg.in_dim)
    return mod