"""TACR critic — separate MLP (the paper's implementation, not a shared
trunk: github.com/VarML/TACR, tac/training/trainer.py defines
Critic([s, a] -> ReLU(512) -> ReLU(256) -> 1) with target networks).

The Phase-3 spec assumed a value head attached to the transformer backbone;
the authors' code uses a standalone critic network. Faithfulness to the
paper wins here — this deviation is documented in PROJECT_NOTES 7.6.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TACRCritic(nn.Module):
    """Q(s, a) MLP — the paper's exact shape (512 -> 256 -> 1)."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        sa = torch.cat([state, action], dim=-1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        return self.l3(q).squeeze(-1)