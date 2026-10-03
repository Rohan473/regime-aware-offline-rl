"""Exposure-only policy under directional uncertainty (H6 follow-up).

Motivation (from the 8.17 diagnosis): direction is at chance, so asking the
policy "up or down?" is asking for noise; the economically consequential
decision is "how much market exposure?".

Model: a_t = e_t  (long-only), with no directional head at all.
  e_t = sigmoid(g(h_t)) in (0,1), learned from the frozen representation.
  s_t is fixed at +1 - the model makes NO directional prediction.

Two variants (pre-specified):
  exposure_sharpe  e_t trained to maximize the differentiable Sharpe of
                   e_t * r_t over the training path (pure risk-adjusted mean).
  exposure_regime  same + penalty rho * E[e_t * risk_t], risk_t = P(bear) +
                   P(crisis) from a train-fit logistic regime head on h_t
                   (the uncertainty/regime-aware exposure control).

Comparators (scripts/rep_exposure.py):
  buy_and_hold (e=1), constant at the learned mean exposure (isolates the
  scale effect from the timing effect), and the current RL policies.

The key comparison is NOT "does it beat buy-and-hold" but: can adaptive
exposure improve the risk/return trade-off WITHOUT requiring directional skill?
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ExposurePolicy(nn.Module):
    def __init__(self, hidden: int, net: int = 64) -> None:
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(hidden, net), nn.ReLU(), nn.Linear(net, 1))

    def e(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.policy(h)).squeeze(-1)


def _sharpe(x: torch.Tensor) -> torch.Tensor:
    m, sd = x.mean(), x.std()
    return m / (sd + 1e-6)


def train_exposure(rep_name: str, cfg, data, risk=None, rho: float = 0.0,
                   epochs: int = 50, lr: float = 1e-3) -> tuple[ExposurePolicy, dict]:
    """Train the exposure-only policy (max Sharpe, optional regime penalty)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    tr, va, te = data.split("train"), data.split("val"), data.split("test")
    H_tr, r_tr = data.H[tr], data.market_returns[tr]
    H_va, r_va = data.H[va], data.market_returns[va]
    risk_tr = torch.tensor(risk.risk(data.H)[tr], dtype=torch.float32) if risk is not None else None

    head = ExposurePolicy(H_tr.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-3)
    best_val, best_state = float("-inf"), None
    for epoch in range(epochs):
        head.train()
        e = head.e(H_tr)
        x = e * r_tr
        loss = -_sharpe(x)
        if risk_tr is not None:
            loss = loss + rho * torch.mean(e * risk_tr)
        if not torch.isfinite(loss).item():
            continue
        opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            xv = head.e(H_va) * r_va
            v = float(_sharpe(xv).item())
        if v > best_val:
            best_val = v
            best_state = {k: v_.clone() for k, v_ in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        e_te = head.e(data.H[te]).numpy()
    r_te = data.market_returns[te].numpy()
    a_te = e_te
    from src.eval.regime_eval import sharpe_ratio
    sr = a_te * r_te
    return head, {
        "rep": rep_name, "seed": cfg.seed, "rho": rho,
        "val_sharpe": float(best_val),
        "test_sharpe": float(sharpe_ratio(sr, 252)),
        "test_exposure": e_te,
        "test_actions": a_te,
        "test_returns": sr,
        "test_mask": te.copy(),
    }