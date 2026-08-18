"""Behavior policies that generate the offline dataset.

Each policy maps a feature vector (and its own internal state) to a scalar
position ``a_t in [position_min, position_max]``. Shorting is enabled by
default ([-1, 1]): -1 = fully short, +1 = fully long, 0 = flat. Every policy
is rolled over the SAME daily series and logs
``(state, action, reward, next_state, done)``; reward is
``a_t * ret_{t+1}`` minus an optional linear transaction cost.

Four policies:
  momentum        long when trailing return > 0, short when < 0, scaled by
                  signal strength (symmetric)
  mean_reversion  contrarian on a short window: short rips, buy dips,
                  scaled by signal strength (symmetric)
  buy_and_hold    constant target allocation
  random          uniform draws inside the position limits (exploration)

Policy trajectories are logged with a ``policy`` tag so later diagnostics
can measure how much each behavior policy contributed to the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from .technical_factors import FEATURE_COLUMNS


@dataclass
class TransitionBuffer:
    """Accumulates (state, action, reward, next_state, done) rows."""

    states: list[np.ndarray] = field(default_factory=list)
    actions: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    next_states: list[np.ndarray] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    dates: list[pd.Timestamp] = field(default_factory=list)
    regimes: list[str] = field(default_factory=list)

    def push(self, state, action, reward, next_state, done, date, regime):
        self.states.append(np.asarray(state, dtype=np.float64))
        self.actions.append(float(action))
        self.rewards.append(float(reward))
        self.next_states.append(np.asarray(next_state, dtype=np.float64))
        self.dones.append(bool(done))
        self.dates.append(date)
        self.regimes.append(regime)

    def to_frame(self, state_cols: list[str], policy_name: str) -> pd.DataFrame:
        if not self.dates:
            raise ValueError(f"policy '{policy_name}' produced zero transitions")
        df = pd.DataFrame(
            {
                "policy": policy_name,
                "date": self.dates,
                "action": self.actions,
                "reward": self.rewards,
                "done": self.dones,
                "regime": self.regimes,
            }
        )
        for j, col in enumerate(state_cols):
            df[col] = [row[j] for row in self.states]
        for j, col in enumerate(state_cols):
            df[f"next_{col}"] = [row[j] for row in self.next_states]
        return df


def _clip_action(a: float, cfg: DictConfig) -> float:
    lo, hi = float(cfg.behavior_policies.position_min), float(cfg.behavior_policies.position_max)
    if lo > hi:
        raise ValueError("position_min must be <= position_max")
    return min(max(a, lo), hi)


def _position_limits(cfg: DictConfig) -> tuple[float, float]:
    lo = float(cfg.behavior_policies.position_min)
    hi = float(cfg.behavior_policies.position_max)
    if lo > hi:
        raise ValueError("position_min must be <= position_max")
    return lo, hi


def _action_cols(cfg: DictConfig) -> list[str]:
    return [f"z_{c}" for c in FEATURE_COLUMNS]


STATE_COLUMNS = [f"z_{c}" for c in FEATURE_COLUMNS]


def momentum_action(state: pd.Series, cfg: DictConfig) -> float:
    """Symmetric momentum: long on positive trailing return, short on
    negative, magnitude scaled by signal strength."""
    window = int(cfg.behavior_policies.policies.momentum.window)
    scale = float(cfg.behavior_policies.policies.momentum.scale)
    ret = state[f"ret_{window}d"]
    if np.isnan(ret):
        raise ValueError("momentum policy saw NaN trailing return; warm-up rows must be dropped upstream")
    lo, hi = _position_limits(cfg)
    return _clip_action(np.clip(ret / scale, lo, hi), cfg)


def mean_reversion_action(state: pd.Series, cfg: DictConfig) -> float:
    """Symmetric contrarian: short rips, buy dips (inverse of momentum on a
    short window), magnitude scaled by signal strength."""
    window = int(cfg.behavior_policies.policies.mean_reversion.window)
    scale = float(cfg.behavior_policies.policies.mean_reversion.scale)
    ret = state[f"ret_{window}d"]
    if np.isnan(ret):
        raise ValueError("mean-reversion policy saw NaN trailing return; warm-up rows must be dropped upstream")
    lo, hi = _position_limits(cfg)
    return _clip_action(np.clip(-ret / scale, lo, hi), cfg)


def buy_and_hold_action(state: pd.Series, cfg: DictConfig) -> float:
    target = float(cfg.behavior_policies.policies.buy_and_hold.target_allocation)
    return _clip_action(target, cfg)


class RandomPolicy:
    """Uniform positions inside [position_min, position_max]; seeded."""

    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def action(self, state: pd.Series, cfg: DictConfig) -> float:
        lo, hi = float(cfg.behavior_policies.position_min), float(cfg.behavior_policies.position_max)
        return float(self._rng.uniform(lo, hi))


def roll_policy(
    features: pd.DataFrame,
    daily_returns: pd.Series,
    regimes: pd.Series,
    policy_name: str,
    cfg: DictConfig,
    seed: int,
) -> pd.DataFrame:
    """Roll one policy over the daily series, logging transitions.

    A transition at date t uses state = features[t], action chosen from
    features[t] only (causal), reward = a_t * ret[t+1] - cost.
    done=True only on the final transition of the trajectory.

    Rows with NaN features or regime are skipped (warm-up), but a transition
    is only emitted when BOTH t and t+1 rows are valid (next_state needed).
    """
    state_cols = _action_cols(cfg)
    random_policy = RandomPolicy(seed) if policy_name == "random" else None
    cost_bps = float(cfg.dataset.transaction_cost_bps)
    cost = cost_bps / 1e4

    valid = features.notna().all(axis=1) & regimes.notna()
    idx = features.index[valid]

    buf = TransitionBuffer()
    for pos in range(len(idx) - 1):
        t, t_next = idx[pos], idx[pos + 1]
        raw_state = features.loc[t, FEATURE_COLUMNS]
        if policy_name == "momentum":
            action = momentum_action(raw_state, cfg)
        elif policy_name == "mean_reversion":
            action = mean_reversion_action(raw_state, cfg)
        elif policy_name == "buy_and_hold":
            action = buy_and_hold_action(raw_state, cfg)
        elif policy_name == "random":
            action = random_policy.action(raw_state, cfg)
        else:
            raise ValueError(f"unknown policy: {policy_name!r}")

        # RL state is the z-scored features (columns state_cols == z_*), NOT raw.
        state = features.loc[t, STATE_COLUMNS].to_numpy()
        next_state = features.loc[t_next, STATE_COLUMNS].to_numpy()
        ret_next = float(daily_returns.loc[t_next])
        if np.isnan(ret_next):
            raise ValueError(f"NaN daily return at {t_next}; pipeline data is corrupt")
        reward = action * ret_next - cost * abs(action)
        done = pos == len(idx) - 2
        buf.push(state, action, reward, next_state, done, t, regimes.loc[t])

    return buf.to_frame(state_cols, policy_name)


def run_all_policies(
    features: pd.DataFrame,
    daily_returns: pd.Series,
    regimes: pd.Series,
    cfg: DictConfig,
) -> pd.DataFrame:
    """Roll every configured behavior policy and concatenate trajectories."""
    seed = int(cfg.behavior_policies.seed)
    names = list(cfg.behavior_policies.policies.keys())
    frames = [
        roll_policy(features, daily_returns, regimes, name, cfg, seed=seed)
        for name in names
    ]
    return pd.concat(frames, ignore_index=True)
