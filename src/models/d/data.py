"""Offline transition data for Model D (IQL).

IQL is a transition-based offline algorithm (unlike Model C's segment-based
decision transformer), so this loader exposes the four Phase-1
behavior-policy trajectories as (s, a, r, s', done) TRANSITIONS plus the
context windows the causal state encoder needs:

- state_t       = the 8 causal z-scored features at decision date t,
                  normalized by TRAIN-split mean/std (same normalization as
                  Model C; val/test stay clean).
- states_in     = the encoder input matrix: 8 raw channels, or (fuzzy
                  variant) 8 raw + 18 IT2-fuzzy memberships = 26. The fuzzy
                  layer is built and FIXED from the TRAIN split only (see
                  fuzzy.py); the memberships are computed ONCE here and the
                  augmented matrix is what the encoder consumes, so the
                  forward pass adds no per-step fuzzy cost.
- action_t      = the policy's logged action in [-1, 1].
- reward_t      = the policy's logged reward (action_t * realized next-day
                  return, matching Phase-1 reward with cost 0).
- done_t        = the dataset's trajectory-end flag (all False within the
                  study horizon: the only True rows are after SPLIT_TEST_END).
- s'            = the state at the NEXT decision date (per-policy states are
                  identical, so s' is the shared next-day feature vector).

Transitions are sampled only from TRAIN-split decision dates whose next-day
transition is genuine (valid mask, same data-hole exclusion as B/C). The
context window for t spans days [t-u+1, t]; the next-state window spans
[t-u+2, t+1]; left-pad (zeros, timestep 0) covers the warm-up before the
series start. Strict causal masking in the encoder makes h_t free of future
information.

The offline dataset may extend past SPLIT_TEST_END after a pipeline re-run;
rows beyond the study horizon are clipped (same as Models B/C).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.loaders import REPO_ROOT
from src.data.technical_factors import FEATURE_COLUMNS
from src.models import SPLIT_TEST_END, SPLIT_TRAIN_END, SPLIT_VAL_END
from src.models.d.config import DConfig, FUZZY_FIELDS
from src.models.d.fuzzy import FuzzyLayer
from src.models.ddr.data import compute_valid_mask

Z_COLUMNS = [f"z_{c}" for c in FEATURE_COLUMNS]
BEHAVIOR_POLICIES = ("momentum", "mean_reversion", "buy_and_hold", "random")

# fuzzy fields -> column index in the 8-d state vector (FEATURE_COLUMNS order)
FUZZY_FIELD_INDICES = [FEATURE_COLUMNS.index(f) for f in FUZZY_FIELDS]


@dataclass
class DData:
    """Per-policy transition arrays + shared market context."""

    states_in: torch.Tensor       # (T, S_in) encoder input (8 or 26-d), train-split normalized
    states_raw: torch.Tensor      # (T, 8) raw normalized states (pre-fuzzy)
    actions: torch.Tensor         # (n_policies, T) logged actions in [-1, 1]
    rewards: torch.Tensor         # (n_policies, T) logged immediate rewards
    dones: torch.Tensor           # (n_policies, T) bool trajectory-end flags
    dates: pd.DatetimeIndex       # tz-aware decision dates (shared across policies)
    regimes: pd.Series            # Phase-1 regime label per date
    valid: torch.Tensor           # (T,) bool; False at data-hole boundaries
    market_returns: torch.Tensor  # (T,) realized next-day market returns
    global_pos: np.ndarray        # (T,) index of each row in the full (pre-split) series
    policies: tuple[str, ...]
    state_mean: np.ndarray        # train-split normalization stats
    state_std: np.ndarray
    fuzzy: FuzzyLayer | None      # the fixed IT2 layer (None for D-minus-fuzzy)


def make_windows(
    states_full: torch.Tensor, ends: np.ndarray, u: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build u-day causal windows ending at each global position.

    states_full: (T, S) full-state matrix. ends: (n,) global end positions.
    Returns (windows (n, u, S), timesteps (n, u) long — absolute positions,
    left-pad rows use timestep 0).
    """
    ends = np.asarray(ends)
    n = len(ends)
    idx = ends[:, None] - (u - 1) + np.arange(u)[None, :]  # (n, u)
    valid = idx >= 0
    idx_clamped = np.clip(idx, 0, None)
    windows = states_full[torch.from_numpy(idx_clamped)] * torch.from_numpy(valid).unsqueeze(-1)
    timesteps = torch.from_numpy(idx_clamped).long()
    return windows, timesteps


def load_d_data(cfg: DConfig, processed_dir: Path | None = None) -> DData:
    """Build the per-policy transitions + fuzzy-augmented encoder input."""
    processed_dir = processed_dir or (REPO_ROOT / "data" / "processed")
    features = pd.read_parquet(processed_dir / "features_regimes.parquet")
    dataset = pd.read_parquet(processed_dir / "offline_dataset.parquet")

    dates = pd.DatetimeIndex(
        sorted(dataset[dataset["policy"].isin(BEHAVIOR_POLICIES)]["date"].unique())
    )
    if len(dates) == 0:
        raise ValueError("no dates found for the behavior policies")
    states_np = features.reindex(dates)[Z_COLUMNS].to_numpy(dtype="float64")

    per_policy = {}
    for pol in BEHAVIOR_POLICIES:
        g = dataset[dataset["policy"] == pol].set_index("date").reindex(dates)
        per_policy[pol] = {
            "actions": g["action"].to_numpy(dtype="float64"),
            "rewards": g["reward"].to_numpy(dtype="float64"),
            "dones": g["done"].to_numpy(dtype="bool"),
        }

    actions_np = np.stack([per_policy[p]["actions"] for p in BEHAVIOR_POLICIES], axis=0)
    rewards_np = np.stack([per_policy[p]["rewards"] for p in BEHAVIOR_POLICIES], axis=0)
    dones_np = np.stack([per_policy[p]["dones"] for p in BEHAVIOR_POLICIES], axis=0)

    regimes = features["regime"].reindex(dates)
    valid = compute_valid_mask(dates, features.index)
    market_ret = features["close"].pct_change().shift(-1)
    market_returns = market_ret.loc[dates].to_numpy(dtype="float64")

    keep = dates.tz_localize(None) <= pd.Timestamp(SPLIT_TEST_END)
    if not keep.all():
        print(
            f"[load_d_data] clipping {int((~keep).sum())} dates beyond {SPLIT_TEST_END}",
            file=sys.stderr,
        )
        dates = dates[keep]
        states_np = states_np[keep]
        actions_np = actions_np[:, keep]
        rewards_np = rewards_np[:, keep]
        dones_np = dones_np[:, keep]
        regimes = regimes[keep]
        valid = valid[keep]
        market_returns = market_returns[keep]

    # train-split normalization (identical to Model C's protocol)
    naive = dates.tz_localize(None)
    train_mask = naive <= pd.Timestamp(SPLIT_TRAIN_END)
    state_mean = states_np[train_mask].mean(axis=0)
    state_std = states_np[train_mask].std(axis=0) + 1e-6
    states_norm = (states_np - state_mean) / state_std

    # fuzzy layer: FIXED memberships initialized from the TRAIN split only
    fuzzy = None
    if cfg.fuzzy:
        fuzzy = FuzzyLayer(
            state_dim=len(FEATURE_COLUMNS),
            field_indices=FUZZY_FIELD_INDICES,
            n_terms=cfg.n_terms,
            mf_width_scale=cfg.mf_width_scale,
            fou_scale=cfg.fou_scale,
        ).set_params_from(states_norm[train_mask])
        states_in = fuzzy(torch.tensor(states_norm, dtype=torch.float32))
    else:
        states_in = torch.tensor(states_norm, dtype=torch.float32)

    return DData(
        states_in=states_in,
        states_raw=torch.tensor(states_norm, dtype=torch.float32),
        actions=torch.tensor(actions_np, dtype=torch.float32),
        rewards=torch.tensor(rewards_np, dtype=torch.float32),
        dones=torch.tensor(dones_np, dtype=torch.bool),
        dates=dates,
        regimes=regimes,
        valid=torch.from_numpy(valid),
        market_returns=torch.tensor(market_returns, dtype=torch.float32),
        global_pos=np.arange(len(dates)),
        policies=BEHAVIOR_POLICIES,
        state_mean=state_mean,
        state_std=state_std,
        fuzzy=fuzzy,
    )


def split_d_data(
    data: DData,
    train_end: str = SPLIT_TRAIN_END,
    val_end: str = SPLIT_VAL_END,
    test_end: str = SPLIT_TEST_END,
) -> dict[str, DData]:
    """Time-ordered split by decision date (identical bounds to B/C)."""
    naive = data.dates.tz_localize(None)
    bounds = {
        "train": naive <= pd.Timestamp(train_end),
        "val": (naive > pd.Timestamp(train_end)) & (naive <= pd.Timestamp(val_end)),
        "test": naive > pd.Timestamp(val_end),
    }
    if naive.max() > pd.Timestamp(test_end):
        raise ValueError(f"dates beyond test_end={test_end}: {naive.max().date()}")

    def sel(mask: np.ndarray) -> DData:
        return DData(
            states_in=data.states_in[mask],
            states_raw=data.states_raw[mask],
            actions=data.actions[:, mask],
            rewards=data.rewards[:, mask],
            dones=data.dones[:, mask],
            dates=data.dates[mask],
            regimes=data.regimes[mask],
            valid=data.valid[mask],
            market_returns=data.market_returns[mask],
            global_pos=data.global_pos[mask],
            policies=data.policies,
            state_mean=data.state_mean,
            state_std=data.state_std,
            fuzzy=data.fuzzy,
        )

    return {name: sel(mask) for name, mask in bounds.items()}


def sample_transitions(
    split: DData, batch_size: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Sample (policy, day t) transitions from a split.

    Only decision dates with a GENUINE next-day transition are sampled
    (split.valid[t] and t+1 exists in the split), matching the data-hole
    exclusion of Models B/C. Returns global positions (into the full states
    matrix), policy indices, and the logged action/reward/done.
    """
    t_len = len(split.dates)
    if t_len < 2:
        raise ValueError("split has no transitionable dates")
    candidate = np.flatnonzero(split.valid.numpy()[: t_len - 1])
    if len(candidate) == 0:
        raise ValueError("split has no valid transitions to sample")
    t_idx = rng.choice(candidate, size=batch_size, replace=True)
    pols = rng.integers(0, len(split.policies), size=batch_size)
    return {
        "pos_t": split.global_pos[t_idx],
        "pos_next": split.global_pos[t_idx + 1],
        "pols": pols,
        "actions": split.actions.numpy()[pols, t_idx],
        "rewards": split.rewards.numpy()[pols, t_idx],
        "dones": split.dones.numpy()[pols, t_idx],
    }


def build_batch(
    split: DData, data_full: DData, cfg: DConfig, rng: np.random.Generator
) -> dict[str, torch.Tensor]:
    """A full IQL training batch: (s_t window, s' window, a, r, done).

    The encoder consumes (B, u, S) windows; s_t and s' are batched together
    into one 2B forward so next-state representations come from the same
    causal encoder with the same parameters.
    """
    tr = sample_transitions(split, cfg.batch_size, rng)
    st, tst = make_windows(data_full.states_in, tr["pos_t"], cfg.u)
    sn, tsn = make_windows(data_full.states_in, tr["pos_next"], cfg.u)
    return {
        "states_t": st,
        "timesteps_t": tst,
        "states_next": sn,
        "timesteps_next": tsn,
        "actions": torch.tensor(tr["actions"], dtype=torch.float32),
        "rewards": torch.tensor(tr["rewards"], dtype=torch.float32),
        "dones": torch.tensor(tr["dones"], dtype=torch.bool),
    }