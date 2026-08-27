"""Behavior policies that generate the offline dataset.

Each policy maps a feature vector (and its own internal state) to a scalar
position ``a_t in [position_min, position_max]``. Shorting is enabled by
default ([-1, 1]): -1 = fully short, +1 = fully long, 0 = flat. Every policy
is rolled over the SAME daily series and logs
``(state, action, reward, next_state, done)``; reward is
``a_t * ret_{t+1}`` minus an optional linear transaction cost.

Policy families:
  momentum        long when trailing return > 0, short when < 0, scaled by
                  signal strength (symmetric). ONE policy per configured
                  window (``momentum_30d``, ``momentum_40d``, ...).
  mean_reversion  contrarian on short windows: short rips, buy dips,
                  scaled by signal strength (symmetric). ONE policy per
                  configured window (``mean_reversion_1d``, ... ``_20d``).
  buy_and_hold    constant target allocation
  random          uniform draws inside the position limits (exploration)
  nr7             SPECIALIZED SIGNAL FAMILY (not window-expanded): the
                  classic NR7 narrow-range breakout, daily-close
                  approximation. A day d is an NR7 day when its high-low
                  range is the narrowest of the 7 trading days ending at d
                  (ties allowed). At decision date t (close), if YESTERDAY
                  (t-1) was an NR7 day and today's close broke OUT of that
                  day's range — close[t] > high[t-1] -> LONG (+1) for
                  t->t+1 (breakout continuation); close[t] < low[t-1] ->
                  SHORT (-1); still inside the range, or no NR7 signal ->
                  FLAT (0). Clean Long/Short/Flat vector; needs the OHLC
                  columns (high/low) of the daily frame. Warm-up: the
                  7-day range minimum must be defined at t-1 (8 rows).

The windowed families compute their trailing log return DIRECTLY from the
daily close (``log(close).diff().rolling(window).sum()``) — the multi-horizon
returns are NOT added to the feature/state frame, so the RL state stays the
8-dimensional Phase-1 feature vector and all downstream model code is
unaffected. Each window is a separate trajectory tagged
``<family>_<N>d``.

Scale normalization: ``scale_N = base_scale * sqrt(N / scale_ref_window)``
(see configs/data.yaml). A uniform scale across windows would shrink fast
windows to near-flat positions and saturate long windows at +/-1; the
sqrt rule keeps the demonstrator's position distribution comparable
because cumulative-return std grows ~ sqrt(N).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from .technical_factors import FEATURE_COLUMNS


def _position_limits(cfg: DictConfig) -> tuple[float, float]:
    lo = float(cfg.behavior_policies.position_min)
    hi = float(cfg.behavior_policies.position_max)
    if lo > hi:
        raise ValueError("position_min must be <= position_max")
    return lo, hi


def _action_cols(cfg: DictConfig) -> list[str]:
    return [f"z_{c}" for c in FEATURE_COLUMNS]


STATE_COLUMNS = [f"z_{c}" for c in FEATURE_COLUMNS]


def _expanded_policy_specs(cfg: DictConfig) -> dict[str, dict]:
    """Expand the configured families into concrete policies.

    momentum / mean_reversion become one policy per window, tagged
    "<family>_<N>d"; scale follows ``base * sqrt(N / scale_ref_window)``.
    Returns {policy_name: {"family", "window", "scale"}}.
    """
    specs: dict[str, dict] = {}
    for name, p in cfg.behavior_policies.policies.items():
        if name == "buy_and_hold":
            specs["buy_and_hold"] = {"family": "buy_and_hold", "window": None, "scale": None}
        elif name == "random":
            specs["random"] = {"family": "random", "window": None, "scale": None}
        elif name == "nr7":
            # specialized signal family: one trajectory, no window expansion
            specs["nr7"] = {"family": "nr7", "window": None, "scale": None}
        elif name in ("momentum", "mean_reversion"):
            base = float(p.scale)
            ref = float(p.scale_ref_window)
            for w in p.windows:
                scale = base * np.sqrt(int(w) / ref)
                specs[f"{name}_{int(w)}d"] = {"family": name, "window": int(w), "scale": scale}
        else:
            raise ValueError(f"unknown behavior policy family: {name!r}")
    return specs


def policy_names(cfg: DictConfig) -> list[str]:
    """The expanded concrete policy names for the configured families."""
    return list(_expanded_policy_specs(cfg).keys())


def roll_policy(
    features: pd.DataFrame,
    daily_returns: pd.Series,
    regimes: pd.Series,
    policy_name: str,
    cfg: DictConfig,
    seed: int,
) -> pd.DataFrame:
    """Roll one policy over the daily series, logging transitions (vectorized).

    A transition at date t uses state = features[t], action chosen from
    features[t] only (causal), reward = a_t * ret[t+1] - cost.
    done=True only on the final transition of the trajectory.

    Rows with NaN features or regime are skipped (warm-up); windowed
    families additionally skip rows before their window's rolling return
    is defined. A transition is only emitted when BOTH t and t+1 rows are
    valid (next_state needed). All per-date arrays are built with pandas
    ``reindex`` + numpy vector ops (no Python loop) so 30+ windowed
    policies roll quickly.
    """
    spec = _expanded_policy_specs(cfg)[policy_name]
    family = spec["family"]
    window = spec["window"]
    state_cols = _action_cols(cfg)
    cost_bps = float(cfg.dataset.transaction_cost_bps)
    cost = cost_bps / 1e4
    lo, hi = _position_limits(cfg)

    if window is not None:
        log_close = np.log(pd.Series(features["close"].to_numpy(dtype="float64"), index=features.index))
        ret_at = log_close.diff().rolling(window).sum()
    else:
        ret_at = None

    valid = features.notna().all(axis=1) & regimes.notna()
    if ret_at is not None:
        valid = valid & ret_at.notna()
    if family == "nr7":
        rng = features["high"] - features["low"]
        # NR7 warm-up: the 7-day range minimum must be defined at t-1
        # (the signal at t looks at YESTERDAY's NR7 status), i.e. rows
        # t-7..t-1 must exist.
        valid = valid & rng.rolling(7).min().shift(1).notna()
    idx = features.index[valid]
    n = len(idx) - 1
    if n <= 0:
        raise ValueError(f"policy '{policy_name}' produced zero transitions")

    t, t_next = idx[:-1], idx[1:]
    states = features.reindex(t)[STATE_COLUMNS].to_numpy(dtype="float64")
    next_states = features.reindex(t_next)[STATE_COLUMNS].to_numpy(dtype="float64")
    ret_next = daily_returns.reindex(t_next).to_numpy(dtype="float64")
    if np.isnan(ret_next).any():
        raise ValueError(
            f"NaN daily return at {t_next[np.isnan(ret_next)][0]}; pipeline data is corrupt"
        )

    if family == "momentum":
        ret = ret_at.reindex(t).to_numpy(dtype="float64")
        if np.isnan(ret).any():
            raise ValueError("momentum policy saw NaN trailing return; warm-up rows must be dropped upstream")
        actions = np.clip(ret / spec["scale"], lo, hi)
    elif family == "mean_reversion":
        ret = ret_at.reindex(t).to_numpy(dtype="float64")
        if np.isnan(ret).any():
            raise ValueError("mean-reversion policy saw NaN trailing return; warm-up rows must be dropped upstream")
        actions = np.clip(-ret / spec["scale"], lo, hi)
    elif family == "buy_and_hold":
        target = float(cfg.behavior_policies.policies.buy_and_hold.target_allocation)
        actions = np.full(n, target)
    elif family == "nr7":
        high = features["high"]
        low = features["low"]
        close = features["close"]
        rng = high - low
        rng_min7 = rng.rolling(7).min()
        # NR7 day: high-low range is the narrowest of the 7 trading days
        # ending at it (ties allowed — inclusive minimum).
        nr7_day = (rng == rng_min7) & rng_min7.notna()
        nr7_prev = nr7_day.shift(1).fillna(False).astype(bool)
        hi_prev = high.shift(1)
        lo_prev = low.shift(1)
        up = nr7_prev & (close > hi_prev)   # upside breakout confirmed at t
        dn = nr7_prev & (close < lo_prev)   # downside breakout confirmed at t
        sig = up.astype("float64") - dn.astype("float64")
        actions = sig.reindex(t).to_numpy(dtype="float64")
        if np.isnan(actions).any():
            raise ValueError("nr7 policy saw NaN signal; warm-up rows must be dropped upstream")
    elif family == "random":
        actions = np.random.default_rng(seed).uniform(lo, hi, size=n)
    else:
        raise ValueError(f"unknown behavior policy: {policy_name!r}")
    actions = np.clip(actions, lo, hi)

    reward = actions * ret_next - cost * np.abs(actions)
    done = np.zeros(n, dtype=bool)
    done[-1] = True

    df = pd.DataFrame(
        {
            "policy": policy_name,
            "date": t,
            "action": actions,
            "reward": reward,
            "done": done,
            "regime": regimes.reindex(t).to_numpy(),
        }
    )
    for j, col in enumerate(state_cols):
        df[col] = states[:, j]
    for j, col in enumerate(state_cols):
        df[f"next_{col}"] = next_states[:, j]
    return df


def run_all_policies(
    features: pd.DataFrame,
    daily_returns: pd.Series,
    regimes: pd.Series,
    cfg: DictConfig,
) -> pd.DataFrame:
    """Roll every configured behavior policy (window-expanded) and
    concatenate trajectories."""
    seed = int(cfg.behavior_policies.seed)
    names = policy_names(cfg)
    frames = [
        roll_policy(features, daily_returns, regimes, name, cfg, seed=seed)
        for name in names
    ]
    return pd.concat(frames, ignore_index=True)
