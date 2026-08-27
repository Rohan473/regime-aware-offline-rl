"""Conditional Gaussian model of the behavior-policy action distribution
p(a | s), used by fix B (the BCQ-style hard constraint).

A lightweight stand-in for BCQ's VAE: the deterministic mode ``mu(s)``
provides the "generated action" the actor's proposal is clipped around.
Fitted by maximum likelihood on the logged (state, action) pairs of all
four behavior policies in the train split, then frozen for the whole
actor-critic run (BCQ trains the generative model first and keeps it
fixed).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ConditionalGaussian(nn.Module):
    """p(a | s) ~ N(mu(s), std); ``forward`` returns the deterministic mode."""

    def __init__(self, state_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Deterministic mode (the 'generated' action) -> (..., 1)."""
        return self.mu(self.net(s))

    def log_prob(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        h = self.net(s)
        mu = self.mu(h)
        std = self.log_std.exp().clamp_min(1e-3)
        var = std * std
        return -0.5 * ((a - mu) ** 2 / var + torch.log(2 * torch.pi * var))

    def nll_loss(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(s, a).mean()


def pretrain_action_model(
    model: ConditionalGaussian,
    split,
    steps: int,
    batch_size: int,
    lr: float,
    rng: np.random.Generator,
    seed: int,
) -> ConditionalGaussian:
    """Fit the generative model to logged (state, action) pairs in a split
    (all behavior policies, decision dates only). Returns the model in eval
    mode with all parameters frozen."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_pol, t_len = split.actions.shape[:2]
    states = split.states.numpy()
    actions = split.actions.numpy()
    for _ in range(steps):
        pols = rng.integers(0, n_pol, size=batch_size)
        ts = rng.integers(0, t_len, size=batch_size)
        s = torch.tensor(states[ts], dtype=torch.float32)
        a = torch.tensor(actions[pols, ts], dtype=torch.float32).unsqueeze(-1)
        opt.zero_grad()
        loss = model.nll_loss(s, a)
        loss.backward()
        opt.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model