"""Regime-aware, uncertainty-constrained offline RL (H5 follow-up).

Research question (from the H4 result that divergence-utility is regime-
dependent): can an offline RL system LEARN when it is safe to deviate from
observed behavior and when it should stay conservative?

Architecture (exactly the proposed pipeline):

  market window -> frozen h_t -> regime/risk estimate -> adaptive divergence
  gate lambda_t -> action  a_t = a_beh_t + lambda_t * d_t

- ``h_t``        : frozen representation (the predictive / contrastive encoder).
- regime estimate: a multinomial logistic head on h_t (train-split fit, no
  look-ahead) giving P(bull), P(bear), P(crisis); risk_t = P(bear)+P(crisis).
- gate lambda_t  : lambda_t = sigmoid(g(h_t)) in (0,1).
- deviation d_t  : d_t = tanh(p(h_t)) in (-1,1).
- action         : a_t = clip(a_beh_t + lambda_t * d_t, -1, 1), where a_beh_t
                   is the per-date behaviour-mean action (the conservative
                   anchor = behaviour cloning).

Training (offline actor-critic, same protocol as the rep x algo matrix):
  - reward_t = a_t * market_return[t]
  - advantage = r_t + gamma*V(h_next) - V(h)
  - policy loss = -logp(g,d|h)*A + value MSE
                 + rho * mean(lambda_t * risk_t)   <-- uncertainty constraint
                 - ent_coef * mean(logp)
  The rho term pushes lambda down exactly in states the regime estimate
  considers risky; in states it considers safe (bull), lambda is free to grow.

Ablations / comparators (in scripts/rep_gated.py):
  - behaviour clone (lambda=0, conservative),
  - unconstrained offline A2C (full deviation),
  - gated with rho=0 (adaptive but unconstrained) vs rho>0 (proposed),
  - oracle gate (lambda=1 in bull, 0.1 otherwise; true test regime labels --
    look-ahead upper bound).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.objective_lab.train import _squashed_logp
from src.models.rep_lab.offline_rl import OfflineRepData, baseline_sharpes

RNG = 42


class RegimeRisk:
    """P(bull/bear/crisis) from h_t, fit on TRAIN only; risk = P(bear)+P(crisis)."""

    def __init__(self) -> None:
        self.sc = None
        self.clf = None

    def fit(self, H: torch.Tensor, regimes: np.ndarray, mask: np.ndarray) -> "RegimeRisk":
        X = H[mask].numpy()
        y = regimes[mask]
        keep = ~np.isin(y, [None, np.nan]) & (y.astype(str) != "nan")
        X, y = X[keep], y[keep]
        self.sc = StandardScaler().fit(X)
        self.clf = LogisticRegression(multi_class="multinomial", max_iter=2000,
                                      random_state=RNG).fit(self.sc.transform(X), y)
        return self

    def proba(self, H: torch.Tensor) -> np.ndarray:
        """(T, 3) P(bull), P(bear), P(crisis) in regime-sorted order."""
        return self.clf.predict_proba(self.sc.transform(H.numpy()))

    def risk(self, H: torch.Tensor) -> np.ndarray:
        """(T,) probability the state is bear or crisis (from h_t only)."""
        p = self.proba(H)
        names = list(self.clf.classes_)
        out = np.zeros(H.shape[0])
        for name, i in zip(names, range(p.shape[1])):
            if name in ("bear", "crisis"):
                out += p[:, i]
        return out


class GatedPolicy(nn.Module):
    """lambda(h), d(h), V(h); action = clip(a_beh + lambda*d, -1, 1)."""

    def __init__(self, hidden: int, net: int = 128) -> None:
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(hidden, net), nn.ReLU(), nn.Linear(net, 2))
        self.value = nn.Sequential(nn.Linear(hidden, net), nn.ReLU(), nn.Linear(net, 1))

    def gd_raw(self, h: torch.Tensor) -> torch.Tensor:
        return self.policy(h)

    def v(self, h: torch.Tensor) -> torch.Tensor:
        return self.value(h).squeeze(-1)

    def deterministic(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(lambda, d) deterministic."""
        g, dr = self.policy(h).split(1, dim=1)
        return torch.sigmoid(g).squeeze(-1), torch.tanh(dr).squeeze(-1)


def compose(a_beh: torch.Tensor, lam: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return (a_beh + lam * d).clamp(-1.0, 1.0)


def _batch_gated(data: OfflineRepData, mask: np.ndarray, n: int, rng: np.random.Generator):
    """Sample t in train; return h_t, h_next, market_ret[t], a_beh[t]."""
    T = data.H.shape[0]
    cand = np.flatnonzero(mask[: T - 1] & data.valid.numpy()[: T - 1])
    t = rng.choice(cand, size=n, replace=True)
    a_beh = data.actions.numpy().mean(axis=0)  # (T,) behaviour-mean action
    return (data.H[t], data.H[t + 1],
            data.market_returns[t].numpy(),
            a_beh[t], t)


def train_gated(rep_name: str, cfg, data: OfflineRepData, risk: RegimeRisk,
                rho: float = 1.0, epochs: int | None = None,
                batch: int = 512, sigma: float = 0.15) -> tuple[GatedPolicy, dict]:
    """Train the gated policy (offline A2C + uncertainty constraint)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    epochs = epochs or cfg.epochs
    head = GatedPolicy(data.H.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr)
    tr_mask, va_mask, te_mask = (data.split("train"), data.split("val"), data.split("test"))
    a_beh_full = data.actions.numpy().mean(axis=0)
    risk_full = risk.risk(data.H)

    def sharpe_of(head, mask):
        with torch.no_grad():
            lam, d = head.deterministic(data.H[mask])
        a = compose(torch.tensor(a_beh_full[mask], dtype=torch.float32), lam, d).numpy()
        r = a * data.market_returns[mask].numpy()
        from src.eval.regime_eval import sharpe_ratio
        return sharpe_ratio(r[np.isfinite(r)], 252)

    best_val, best_state = float("-inf"), None
    for epoch in range(epochs):
        head.train()
        rng = np.random.default_rng(cfg.seed + epoch)
        n_iter = max(1, int(tr_mask.sum()) // batch)
        for _ in range(n_iter):
            h, h_next, r_t, a_beh_t, t = _batch_gated(data, tr_mask, batch, rng)
            h = h.float(); h_next = h_next.float()
            r_t = torch.tensor(r_t, dtype=torch.float32)
            a_beh_t = torch.tensor(a_beh_t, dtype=torch.float32)
            risk_t = torch.tensor(risk_full[t], dtype=torch.float32)

            gd = head.gd_raw(h)
            g, dr = gd.split(1, dim=1)
            lam = torch.sigmoid(g)
            d = torch.tanh(dr)
            # Gaussian in (g, dr) space for logp of the sampled action
            eps = torch.randn_like(gd) * sigma
            gd_s = gd + eps
            g_s, dr_s = gd_s.split(1, dim=1)
            lam_s = torch.sigmoid(g_s)
            d_s = torch.tanh(dr_s)
            logp = (_squashed_logp(g_s, g, sigma, lam_s.squeeze(-1))
                    + _squashed_logp(dr_s, dr, sigma, d_s.squeeze(-1)))

            a = compose(a_beh_t, lam_s.squeeze(-1), d_s.squeeze(-1))
            with torch.no_grad():
                v_next = head.v(h_next)
            v_t = head.v(h)
            td = r_t * a + cfg.gamma * v_next
            adv = (td - v_t).detach()
            loss = (-torch.mean(logp * adv) - cfg.ent_coef * torch.mean(logp)
                    + torch.mean((v_t - td.detach()) ** 2)
                    + rho * torch.mean(lam.squeeze(-1) * risk_t))
            if not torch.isfinite(loss).item():
                continue
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        v = sharpe_of(head, va_mask)
        if np.isfinite(v) and v > best_val:
            best_val = v
            best_state = {k: v_.clone() for k, v_ in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    te_sharpe = sharpe_of(head, te_mask)
    with torch.no_grad():
        lam_te, d_te = head.deterministic(data.H[te_mask])
    a_te = compose(torch.tensor(a_beh_full[te_mask], dtype=torch.float32), lam_te, d_te).numpy()
    r_te = a_te * data.market_returns[te_mask].numpy()
    return head, {
        "rep": rep_name, "seed": cfg.seed, "rho": rho,
        "val_sharpe": float(best_val), "test_sharpe": float(te_sharpe),
        "test_lambda_mean": float(lam_te.mean().item()),
        "test_lambda": lam_te.numpy(),
        "test_actions": a_te,
        "test_returns": r_te,
        "test_mask": te_mask.copy(),
        "test_return": float(np.prod(1 + r_te) - 1) if r_te.size else float("nan"),
    }