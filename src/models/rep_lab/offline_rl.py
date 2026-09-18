"""Offline RL on FROZEN representations: representation x {A2C, IQL}.

The rep x policy lab (section 8.5) trained a single A2C head on the market
path with self-generated actions. This module instead puts BOTH algorithms on
the SAME offline transition dataset (the Phase-1 behavior trajectories), so
the representation is the only thing that varies and the algorithm is the
second factor:

  A2C  advantage actor-critic: Gaussian policy on h_t, critic V(h_t),
       advantage A = r + gamma V(h') - V(h), loss = -logpi(a_logged|h) A
       + value MSE. (Offline A2C: actions are the LOGGED behavior actions.)
  IQL  Implicit Q-Learning (Kostrikov et al. 2022): expectile V, Q(h,a) with
       target nets, and AWR policy extraction exp(beta*(Q-V)) (pi - a)^2.
       No query of Q on out-of-distribution actions.

Both consume the same frozen h_t (encoder never updated) and are evaluated
identically: deterministic policy action * realized market return on the TEST
split -> Sharpe. This isolates "does the decision layer matter, given a
representation?" -- the representation-quality != trading-performance test.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.eval.regime_eval import max_drawdown, sharpe_ratio
from src.models import SPLIT_TEST_END, SPLIT_TRAIN_END, SPLIT_VAL_END
from src.models.objective_lab.train import _squashed_logp
from src.models.rep_lab.config import RepLabConfig, tag_path
from src.models.rep_lab.train import BEST, load_rep

ALGOS = ("BC", "A2C", "IQL", "CQL")


# --------------------------------------------------------------------------
# offline transition dataset on the frozen representation
# --------------------------------------------------------------------------

@dataclass
class OfflineRepData:
    H: torch.Tensor              # (T, hidden) frozen representation per date
    actions: torch.Tensor        # (P, T) logged behavior actions
    rewards: torch.Tensor        # (P, T) logged rewards
    dones: torch.Tensor          # (P, T) bool
    market_returns: torch.Tensor  # (T,)
    valid: torch.Tensor          # (T,) bool
    dates: pd.DatetimeIndex
    policies: tuple[str, ...]

    def split(self, name: str) -> np.ndarray:
        naive = self.dates.tz_localize(None)
        if name == "train":
            return np.asarray(naive <= pd.Timestamp(SPLIT_TRAIN_END))
        if name == "val":
            return np.asarray((naive > pd.Timestamp(SPLIT_TRAIN_END))
                              & (naive <= pd.Timestamp(SPLIT_VAL_END)))
        return np.asarray(naive > pd.Timestamp(SPLIT_VAL_END))


def load_offline_rep(rep_name: str, cfg: RepLabConfig,
                     policies: tuple[str, ...] | None = None,
                     max_policies: int | None = None) -> OfflineRepData:
    """Frozen per-date H + logged (a, r, done) transitions from Phase-1.

    H comes from ``load_rep_data`` (canonical 8, or ``cfg.feature_cols``);
    the logged transitions come from ``offline_dataset.parquet`` on the
    dates common to every selected policy (shared-state design)."""
    from src.data.loaders import REPO_ROOT
    from src.models.rep_lab.data import load_rep_data

    dataset = pd.read_parquet(REPO_ROOT / "data" / "processed" / "offline_dataset.parquet")
    if policies is None:
        policies = tuple(sorted(dataset["policy"].unique()))
    if max_policies is not None and len(policies) > max_policies:
        rng = np.random.RandomState(cfg.seed)
        policies = tuple(sorted(rng.choice(list(policies), max_policies, replace=False)))

    date_sets = [set(dataset[dataset["policy"] == p]["date"].unique()) for p in policies]
    common = pd.DatetimeIndex(sorted(set.intersection(*date_sets)))

    base = load_rep_data(cfg)
    # the rep-lab path is clipped to SPLIT_TEST_END, so drop any policy dates
    # beyond it before aligning (the offline dataset may extend further)
    pos = base.dates.get_indexer(common)
    keep = pos >= 0
    common, pos = common[keep], pos[keep]
    if len(common) == 0:
        raise ValueError("no policy dates overlap the rep-lab path")
    order = np.argsort(pos)
    keep_dates = base.dates[pos[order]]
    H = encode_rep(rep_name, cfg, base.windows[pos[order]])
    common = common[order]

    acts = np.stack([
        dataset[dataset["policy"] == p].set_index("date").reindex(common)["action"]
        .to_numpy(dtype="float64") for p in policies])
    rews = np.stack([
        dataset[dataset["policy"] == p].set_index("date").reindex(common)["reward"]
        .to_numpy(dtype="float64") for p in policies])
    dones = np.stack([
        dataset[dataset["policy"] == p].set_index("date").reindex(common)["done"]
        .to_numpy(dtype="bool") for p in policies])

    return OfflineRepData(
        H=H,
        actions=torch.tensor(acts, dtype=torch.float32),
        rewards=torch.tensor(rews, dtype=torch.float32),
        dones=torch.tensor(dones, dtype=torch.bool),
        market_returns=base.next_returns[pos[order]],
        valid=base.valid[pos[order]],
        dates=keep_dates,
        policies=policies,
    )


def encode_rep(rep_name: str, cfg: RepLabConfig, windows: torch.Tensor) -> torch.Tensor:
    """Frozen per-date feature for 'raw' (flat) or a learned objective.

    ``cfg.tag`` selects a sweep variant of that objective when set."""
    if rep_name == "raw":
        return windows.reshape(windows.shape[0], -1)
    rep = load_rep(rep_name, cfg, tag_path(cfg, rep_name) / BEST)
    with torch.no_grad():
        return rep.encoder.encode(windows)


# --------------------------------------------------------------------------
# heads
# --------------------------------------------------------------------------

class A2CHead(nn.Module):
    def __init__(self, hidden: int, net: int = 128) -> None:
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(hidden, net), nn.ReLU(), nn.Linear(net, 1))
        self.value = nn.Sequential(nn.Linear(hidden, net), nn.ReLU(), nn.Linear(net, 1))

    def mu_raw(self, h): return self.policy(h).squeeze(-1)
    def mu(self, h): return torch.tanh(self.mu_raw(h))
    def v(self, h): return self.value(h).squeeze(-1)


class BCHead(nn.Module):
    """Behavior cloning: deterministic policy regressed onto logged actions."""

    def __init__(self, hidden: int, net: int = 128) -> None:
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(hidden, net), nn.ReLU(), nn.Linear(net, 1))

    def mu(self, h): return torch.tanh(self.policy(h).squeeze(-1))


class CQLHead(nn.Module):
    """Conservative Q-Learning (CQL(H)) with a deterministic actor.

    Q(h,a) trained on the Bellman target r + gamma V(s'), where the soft value
    V(s') = logsumexp_a Q_target(h',a) - log K over K sampled actions, plus the
    conservative penalty alpha * (logsumexp_a Q(h,a) - Q(h,a_logged)). The
    actor maximises Q. This is the standard continuous-action CQL(H) form."""

    def __init__(self, hidden: int, net: int = 128, n_actions: int = 10,
                 alpha: float = 1.0, bc_coef: float = 0.5,
                 tau: float = 0.005) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.alpha = alpha
        self.bc_coef = bc_coef  # TD3+BC-style actor regularizer (anti-saturation)
        self.tau = tau
        self.q_head = nn.Sequential(nn.Linear(hidden + 1, net), nn.ReLU(), nn.Linear(net, 1))
        self.policy = nn.Sequential(nn.Linear(hidden, net), nn.ReLU(), nn.Linear(net, 1))
        self._q_target = copy.deepcopy(self.q_head)
        for p in self._q_target.parameters():
            p.requires_grad_(False)

    def q(self, h, a):
        return self.q_head(torch.cat((h, a.unsqueeze(-1)), dim=-1)).squeeze(-1)

    def q_t(self, h, a):
        return self._q_target(torch.cat((h, a.unsqueeze(-1)), dim=-1)).squeeze(-1)

    def mu(self, h): return torch.tanh(self.policy(h).squeeze(-1))

    def polyak(self) -> None:
        with torch.no_grad():
            for p, tp in zip(self.q_head.parameters(), self._q_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)


class IQLHead(nn.Module):
    def __init__(self, hidden: int, net: int = 128, tau: float = 0.005) -> None:
        super().__init__()
        self.tau = tau
        self.v_head = nn.Sequential(nn.Linear(hidden, net), nn.ReLU(), nn.Linear(net, 1))
        self.q_head = nn.Sequential(nn.Linear(hidden + 1, net), nn.ReLU(), nn.Linear(net, 1))
        self.policy = nn.Sequential(nn.Linear(hidden, net), nn.ReLU(), nn.Linear(net, 1))
        self._v_target = copy.deepcopy(self.v_head)
        self._q_target = copy.deepcopy(self.q_head)
        for p in (*self._v_target.parameters(), *self._q_target.parameters()):
            p.requires_grad_(False)

    def v(self, h): return self.v_head(h).squeeze(-1)
    def q(self, h, a): return self.q_head(torch.cat((h, a.unsqueeze(-1)), dim=-1)).squeeze(-1)
    def v_t(self, h): return self._v_target(h).squeeze(-1)
    def q_t(self, h, a):
        return self._q_target(torch.cat((h, a.unsqueeze(-1)), dim=-1)).squeeze(-1)
    def mu(self, h): return torch.tanh(self.policy(h).squeeze(-1))

    def polyak(self) -> None:
        with torch.no_grad():
            for p, tp in zip(self.v_head.parameters(), self._v_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.q_head.parameters(), self._q_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)


def expectile_loss(pred, target, tau: float) -> torch.Tensor:
    diff = (target - pred).detach()
    w = torch.where(diff < 0, 1.0 - tau, tau)
    return (w * (target - pred) ** 2).mean()


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def _batch(data: OfflineRepData, mask: np.ndarray, n: int, rng: np.random.Generator):
    T = data.H.shape[0]
    cand = np.flatnonzero(mask[: T - 1] & data.valid.numpy()[: T - 1])
    if len(cand) == 0:
        raise ValueError("no valid transitions in split")
    t = rng.choice(cand, size=n, replace=True)
    p = rng.integers(0, data.actions.shape[0], size=n)
    return (data.H[t], data.H[t + 1],
            data.actions.numpy()[p, t], data.rewards.numpy()[p, t],
            data.dones.numpy()[p, t])


def baseline_sharpes(data: OfflineRepData) -> dict:
    """Trivial decision baselines on the test split (no learning).

    buy_and_hold      constant +1 position
    constant_mean     the single mean logged action across all transitions
    behavior_mean     the per-date mean logged action (action-frequency mix)
    random            seeded uniform actions in [-1, 1]
    """
    te = data.split("test")
    mr = data.market_returns[te].numpy()
    acts = data.actions[:, te].numpy()
    rng = np.random.default_rng(0)
    return {
        "buy_and_hold": float(sharpe_ratio(mr, 252)),
        "constant_mean": float(sharpe_ratio(np.full_like(mr, acts.mean()) * mr, 252)),
        "behavior_mean": float(sharpe_ratio(acts.mean(axis=0) * mr, 252)),
        "random": float(sharpe_ratio(rng.uniform(-1, 1, mr.shape) * mr, 252)),
    }


def _eval_sharpe(head, data: OfflineRepData, mask: np.ndarray) -> float:
    with torch.no_grad():
        a = head.mu(data.H[mask]).numpy()
    r = a * data.market_returns[mask].numpy()
    r = r[np.isfinite(r)]
    return sharpe_ratio(r, periods_per_year=252)


def train_offline(rep_name: str, algo: str, cfg: RepLabConfig,
                  data: OfflineRepData, epochs: int | None = None,
                  batch: int = 512) -> tuple[nn.Module, dict]:
    """Train BC / A2C / IQL / CQL on the frozen representation; (head, metrics)."""
    if algo not in ALGOS:
        raise ValueError(f"algo must be one of {ALGOS}")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    epochs = epochs or cfg.epochs
    hidden = data.H.shape[1]
    if algo == "CQL":
        head = CQLHead(hidden, bc_coef=cfg.cql_bc_coef)
    else:
        head = {"BC": BCHead, "A2C": A2CHead, "IQL": IQLHead}[algo](hidden)
    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr)
    tr_mask, va_mask, te_mask = (data.split("train"), data.split("val"), data.split("test"))

    beta, iql_tau = 3.0, 0.7
    best_val, best_state = float("-inf"), None
    for epoch in range(epochs):
        rng = np.random.default_rng(cfg.seed + epoch)
        head.train()
        n_iter = max(1, int(tr_mask.sum()) // batch)
        for _ in range(n_iter):
            h, h_next, a, r, done = _batch(data, tr_mask, batch, rng)
            h = h.float(); h_next = h_next.float()
            a = torch.tensor(a, dtype=torch.float32)
            r = torch.tensor(r, dtype=torch.float32)
            done = torch.tensor(done, dtype=torch.bool)
            if algo == "BC":
                loss = torch.mean((head.mu(h) - a) ** 2)
            elif algo == "A2C":
                with torch.no_grad():
                    v_next = head.v(h_next)
                v_t = head.v(h)
                mu_raw = head.mu_raw(h)
                a_c = a.clamp(-0.999, 0.999)
                pt = torch.atanh(a_c)
                logp = _squashed_logp(pt, mu_raw, cfg.ac_sigma, a_c)
                td = r + cfg.gamma * v_next * (~done)
                adv = (td - v_t).detach()
                loss = (-torch.mean(logp * adv) - cfg.ent_coef * torch.mean(logp)
                        + torch.mean((v_t - td.detach()) ** 2))
            elif algo == "IQL":
                with torch.no_grad():
                    q_target = head.q_t(h, a)
                    v_target = head.v_t(h_next)
                    td = r + cfg.gamma * v_target * (~done)
                loss = (expectile_loss(head.v(h), q_target, iql_tau)
                        + torch.mean((head.q(h, a) - td) ** 2))
                with torch.no_grad():
                    adv = head.q(h, a) - head.v(h)
                    w = torch.exp(beta * adv).clamp(max=100.0)
                loss = loss + torch.mean(w * (head.mu(h) - a) ** 2)
            else:  # CQL(H)
                k = head.n_actions
                rand_a = torch.rand(k * h.shape[0], 1, generator=None) * 2 - 1
                rand_h = h.repeat(k, 1)
                q_rand = head.q(rand_h, rand_a.squeeze(-1)).view(k, -1)
                with torch.no_grad():
                    rand_h2 = h_next.repeat(k, 1)
                    q_next = head.q_t(rand_h2, rand_a.squeeze(-1)).view(k, -1)
                    soft_v_next = torch.logsumexp(q_next, dim=0) - np.log(k)
                    td = r + cfg.gamma * soft_v_next * (~done)
                q_sa = head.q(h, a)
                logsumexp_sa = torch.logsumexp(q_rand, dim=0) - np.log(k)
                cons = torch.mean(logsumexp_sa - q_sa)
                loss = torch.mean((q_sa - td) ** 2) + head.alpha * cons
                # actor: maximise Q, regularised toward the logged action
                loss = (loss - torch.mean(head.q(h, head.mu(h)))
                        + head.bc_coef * torch.mean((head.mu(h) - a) ** 2))
            if not torch.isfinite(loss).item():
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            if algo in ("IQL", "CQL"):
                head.polyak()

        head.eval()
        v = _eval_sharpe(head, data, va_mask)
        if np.isfinite(v) and v > best_val:
            best_val = v
            best_state = copy.deepcopy(head.state_dict())
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    te_sharpe = _eval_sharpe(head, data, te_mask)
    with torch.no_grad():
        a_test = head.mu(data.H[te_mask]).numpy()
    r_test = a_test * data.market_returns[te_mask].numpy()
    r_test = r_test[np.isfinite(r_test)]
    return head, {
        "algo": algo, "rep": rep_name, "seed": cfg.seed,
        "val_sharpe": float(best_val), "test_sharpe": float(te_sharpe),
        "test_return": float(np.prod(1 + r_test) - 1) if r_test.size else float("nan"),
        "test_maxdd": float(max_drawdown(r_test)) if r_test.size else float("nan"),
        "test_turnover": float(np.abs(np.diff(a_test, prepend=0.0)).mean()) if a_test.size else float("nan"),
    }
