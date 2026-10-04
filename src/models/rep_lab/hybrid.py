"""Hybrid exposure policy (H7 follow-up).

Research question: can adaptive exposure preserve the risk-timing benefit of
offline RL while recovering the persistent market exposure that buy-and-hold
provides when directional skill is unavailable?

Continuous blend (no arbitrary threshold, no discontinuous trading):

    a_t = e_t + alpha_t * (1 - e_t),    alpha_t in [0,1]

  alpha = 0 -> pure adaptive exposure e_t
  alpha = 1 -> buy-and-hold (full market exposure)

alpha_t is learned from EXPLICIT risk features available at t (no future
returns/volatility, no direction):
    risk_t   = P(bear) + P(crisis)        (train-fit regime head on h_t)
    ood_t    = Mahalanobis distance of h_t to the train representation
    vol_t    = current realized volatility (z-scored)

alpha_t = sigmoid(linear([risk, ood, vol])) -- a 3-parameter head, so the
loading signs are directly readable: does low risk / in-distribution /
low-vol push alpha toward 1 (approach buy-and-hold)?

Baselines in scripts/rep_hybrid.py: buy-and-hold (1), adaptive e_t, constant
mean exposure, 50/50 blend, floor max(e, tau), and the proposed risk-aware
blend. Metrics: Sharpe, return, max drawdown, exposure, upside capture,
downside loss, regime exposure -- across SPY / CSI300 / NIFTY.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class BlendPolicy(nn.Module):
    """alpha_t = sigmoid(linear([risk, ood, vol]))."""

    def __init__(self, n_feat: int = 3) -> None:
        super().__init__()
        self.head = nn.Linear(n_feat, 1)

    def alpha(self, F: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(F)).squeeze(-1)


def compose(e: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """a = e + alpha*(1-e), monotone in e and alpha; stays in [0,1]."""
    return e + alpha * (1.0 - e)


def _sharpe(x: torch.Tensor) -> torch.Tensor:
    return x.mean() / (x.std() + 1e-6)


def train_blend(e_train, F_train, r_train, e_val, F_val, r_val,
                seed: int, epochs: int = 200, lr: float = 1e-2,
                weight_decay: float = 1e-3) -> tuple[BlendPolicy, float]:
    """Train the alpha head (frozen e) to maximise Sharpe of a*r."""
    torch.manual_seed(seed)
    mu, sd = F_train.mean(0), F_train.std(0) + 1e-8
    Fs_train = (F_train - mu) / sd
    Fs_val = (F_val - mu) / sd

    head = BlendPolicy(F_train.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state = float("-inf"), None
    for epoch in range(epochs):
        head.train()
        a = compose(e_train, head.alpha(Fs_train))
        loss = -_sharpe(a * r_train)
        if not torch.isfinite(loss).item():
            continue
        opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            av = compose(e_val, head.alpha(Fs_val))
            v = float(_sharpe(av * r_val).item())
        if v > best_val:
            best_val = v
            best_state = {k: v_.clone() for k, v_ in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head, float(best_val)