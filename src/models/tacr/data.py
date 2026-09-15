"""Offline trajectory data for Model C (TACR).

Decision-Transformer-style models are BUILT for offline trajectory data, so
this loader uses the four Phase-1 behavior-policy trajectories DIRECTLY (the
natural fit, per the Phase-3 spec): each policy is one continuous trajectory
of (state_t, action_t, reward_t) over the full market path.

Per trajectory (policy):
- state_t   = the 8 causal z-scored features at decision date t (identical
              to the LAST row of Model B's W-day window at t — the alignment
              test in tests/test_tacr.py pins this).
- action_t  = the policy's logged action in [-1, 1].
- reward_t  = the policy's logged reward (action_t * realized next-day
              return, matching Phase-1 reward with cost 0).
- rtg_t     = return-to-go: cumulative FUTURE logged reward from t (per the
              Phase-3 spec: "compute from the offline dataset's logged
              reward series per behavior-policy trajectory").
- timestep  = absolute position within the trajectory (embedded like a
              positional encoding, per the paper's GPT-2-without-positions
              design).

Splits and alignment:
- Time-based train/val/test by decision date, identical bounds to Model B
  (src.models.SPLIT_*_END) — the B/C comparison is on the same calendar.
- Context at decision date t spans days [t-u+1, t] (past + current only;
  strict causal mask in the policy), so no label leakage across splits —
  the same causality argument as Model B's trailing windows.
- State normalization uses TRAIN-split mean/std only (the paper normalizes
  over the whole file, which mixes val/test — flagged in PROJECT_NOTES 7.6).
- RTGs at train dates are computed over the full logged trajectory and may
  include rewards beyond the train split: the return-to-go is the
  CONDITIONING target (a goal channel), not an input feature, and the
  val/test rolls use the constant ``rtg_target`` — no realized future
  information enters evaluation metrics.
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
from src.models.ddr.data import compute_valid_mask

Z_COLUMNS = [f"z_{c}" for c in FEATURE_COLUMNS]


def behavior_policy_tags(
    available: tuple[str, ...], exclude_policies: tuple[str, ...] = ()
) -> tuple[str, ...]:
    """Effective TACR policy set: ``available`` minus excluded tags,
    preserving order."""
    excluded = set(exclude_policies)
    return tuple(p for p in available if p not in excluded)


@dataclass
class TACRData:
    """Merged behavior-policy trajectories (states shared across policies)."""

    states: torch.Tensor        # (T, F) normalized day-t feature vectors (identical across policies)
    actions: torch.Tensor       # (n_policies, T) logged actions in [-1, 1]
    rewards: torch.Tensor       # (n_policies, T) logged immediate rewards
    rtgs: torch.Tensor          # (n_policies, T) return-to-go (cumulative future reward)
    timesteps: torch.Tensor     # (n_policies, T) long, absolute position in trajectory
    dates: pd.DatetimeIndex     # tz-aware decision dates (shared across policies)
    regimes: pd.Series          # Phase-1 regime label per date
    valid: torch.Tensor         # (T,) bool; False at data-hole boundaries
    market_returns: torch.Tensor  # (T,) realized next-day market returns
    policies: tuple[str, ...]
    state_mean: np.ndarray      # train-split normalization stats
    state_std: np.ndarray


def load_tacr_data(
    cfg_u: int | None = None,
    processed_dir: Path | None = None,
    policies: tuple[str, ...] | None = None,
    exclude_policies: tuple[str, ...] = (),
    test_end: str | None = None,
    state_macro: bool = False,
) -> TACRData:
    """Build the behavior-policy trajectories from Phase-1 artifacts.

    ``cfg_u`` is unused for loading (kept for a symmetric signature with
    Model B); context windows are cut at batch time from these trajectories.
    ``policies`` defaults to ALL policy tags present in the offline dataset
    (the window-expanded behavior families), minus ``exclude_policies``
    (e.g. ``("random",)`` to drop the uniform demonstrator).
    ``test_end`` overrides the study-horizon clip (the default
    ``SPLIT_TEST_END``) — used by the 7.26 out-of-sample confirmation run to
    load the untouched 2025-2026 window for a frozen-artifact roll.
    """
    from src.models.tacr.config import TACRConfig

    u = cfg_u if cfg_u is not None else TACRConfig().u
    processed_dir = processed_dir or (REPO_ROOT / "data" / "processed")
    features = pd.read_parquet(processed_dir / "features_regimes.parquet")
    dataset = pd.read_parquet(processed_dir / "offline_dataset.parquet")

    if policies is None:
        policies = tuple(
            sorted(set(dataset["policy"].unique()) - set(exclude_policies))
        )
    # policies have DIFFERENT date coverage (windowed families start after
    # their lookback warm-up), so only dates present in EVERY selected
    # policy's trajectory are usable: the shared-state design needs each
    # policy to have a logged transition on every date.
    date_sets = [
        set(dataset[dataset["policy"] == pol]["date"].unique()) for pol in policies
    ]
    if any(len(s) == 0 for s in date_sets):
        missing = [pol for pol, s in zip(policies, date_sets) if len(s) == 0]
        raise ValueError(f"policies with no logged transitions: {missing}")
    common = sorted(set.intersection(*date_sets))
    dates = pd.DatetimeIndex(common)
    if len(dates) == 0:
        raise ValueError("no dates found for the behavior policies")

    states_base = features.reindex(dates)[Z_COLUMNS].to_numpy(dtype="float64")

    # Exp 1 (PROJECT_NOTES 7.30.1): optional 16-dim state. Prepend the macro
    # z-features and restrict dates AT LOAD TIME to rows where every macro
    # feature is finite (2007-05+). The SPY z-columns stay first so the first
    # 8 state dims are bit-identical to the canonical 8-dim run — only 8 new
    # (macro) dims are appended, keeping the expansion purely additive.
    if state_macro:
        from src.data.loaders import REPO_ROOT as _ROOT  # noqa: F811
        from src.data.macro_factors import MACRO_FEATURES, macro_features, zscore_causal  # noqa: F401

        feats_full = pd.read_parquet(_ROOT / "data" / "processed" / "features_regimes.parquet")
        naive = dates.tz_localize(None)
        macro_raw = macro_features(feats_full)
        macro_z = np.column_stack(
            [zscore_causal(macro_raw[c]) for c in MACRO_FEATURES]
        )
        macro_z = macro_z[np.searchsorted(macro_raw.index, naive)]      # (T, 8)
        base_ok = np.isfinite(states_base).all(1)
        mac_ok = np.isfinite(macro_z).all(1)
        keep = base_ok & mac_ok
        if not keep.all():
            dropped = int((~keep).sum())
            print(f"[load_tacr_data] state_macro=True: dropping {dropped} dates "
                  f"without finite macro features (pre-2007-05 + gaps)",
                  file=sys.stderr)
        dates = dates[keep]
        states_np = np.concatenate(
            [states_base[keep], macro_z[keep].astype(np.float64, copy=False)], axis=1
        )
    else:
        states_np = states_base

    per_policy = {}
    for pol in policies:
        g = dataset[dataset["policy"] == pol].set_index("date").reindex(dates)
        per_policy[pol] = {
            "actions": g["action"].to_numpy(dtype="float64"),
            "rewards": g["reward"].to_numpy(dtype="float64"),
        }

    # return-to-go per trajectory: cumulative FUTURE logged reward (computed
    # after the study-horizon clip so rtgs == cumsum of stored rewards)
    actions_np = np.stack([per_policy[p]["actions"] for p in policies], axis=0)
    rewards_np = np.stack([per_policy[p]["rewards"] for p in policies], axis=0)

    regimes = features["regime"].reindex(dates)
    valid = compute_valid_mask(dates, features.index)
    market_ret = features["close"].pct_change().shift(-1)
    market_returns = market_ret.loc[dates].to_numpy(dtype="float64")

    keep = dates.tz_localize(None) <= pd.Timestamp(test_end or SPLIT_TEST_END)
    if not keep.all():
        clip_end = test_end or SPLIT_TEST_END
        print(
            f"[load_tacr_data] clipping {int((~keep).sum())} dates beyond {clip_end}",
            file=sys.stderr,
        )
        dates = dates[keep]
        states_np = states_np[keep]
        actions_np = actions_np[:, keep]
        rewards_np = rewards_np[:, keep]
        regimes = regimes[keep]
        valid = valid[keep]
        market_returns = market_returns[keep]

    rtgs_np = np.stack(
        [np.cumsum(rewards_np[p][::-1])[::-1] for p in range(len(policies))], axis=0
    )

    # train-split normalization stats (paper normalizes over the whole file;
    # we restrict to train to keep val/test clean)
    naive = dates.tz_localize(None)
    train_mask = naive <= pd.Timestamp(SPLIT_TRAIN_END)
    state_mean = states_np[train_mask].mean(axis=0)
    state_std = states_np[train_mask].std(axis=0) + 1e-6
    states_norm = (states_np - state_mean) / state_std

    timesteps = np.arange(len(dates))[None, :].repeat(len(policies), axis=0)

    return TACRData(
        states=torch.tensor(states_norm, dtype=torch.float32),
        actions=torch.tensor(actions_np, dtype=torch.float32),
        rewards=torch.tensor(rewards_np, dtype=torch.float32),
        rtgs=torch.tensor(rtgs_np, dtype=torch.float32),
        timesteps=torch.tensor(timesteps, dtype=torch.long),
        dates=dates,
        regimes=regimes,
        valid=torch.from_numpy(valid),
        market_returns=torch.tensor(market_returns, dtype=torch.float32),
        policies=policies,
        state_mean=state_mean,
        state_std=state_std,
    )


def split_tacr_data(
    data: TACRData,
    train_end: str = SPLIT_TRAIN_END,
    val_end: str = SPLIT_VAL_END,
    test_end: str = SPLIT_TEST_END,
) -> dict[str, TACRData]:
    """Time-ordered split by decision date (identical bounds to Model B)."""
    naive = data.dates.tz_localize(None)
    bounds = {
        "train": naive <= pd.Timestamp(train_end),
        "val": (naive > pd.Timestamp(train_end)) & (naive <= pd.Timestamp(val_end)),
        "test": naive > pd.Timestamp(val_end),
    }
    if naive.max() > pd.Timestamp(test_end):
        raise ValueError(f"dates beyond test_end={test_end}: {naive.max().date()}")

    def sel(mask: np.ndarray) -> TACRData:
        return TACRData(
            states=data.states[mask],
            actions=data.actions[:, mask],
            rewards=data.rewards[:, mask],
            rtgs=data.rtgs[:, mask],
            timesteps=data.timesteps[:, mask],
            dates=data.dates[mask],
            regimes=data.regimes[mask],
            valid=data.valid[mask],
            market_returns=data.market_returns[mask],
            policies=data.policies,
            state_mean=data.state_mean,
            state_std=data.state_std,
        )

    return {name: sel(mask) for name, mask in bounds.items()}


def _family_of(policy: str) -> str:
    """Family root of a policy tag: ``momentum_30d`` -> ``momentum``,
    ``mean_reversion_10d`` -> ``mean_reversion``, ``buy_and_hold`` /
    ``random`` -> themselves."""
    if policy in ("buy_and_hold", "random"):
        return policy
    return policy.rsplit("_", 1)[0]


def sample_batch(
    split: TACRData,
    u: int,
    batch_size: int,
    rng: np.random.Generator,
    balanced_families: bool = False,
) -> dict[str, torch.Tensor]:
    """Sample u-length segments fully inside the split (decision dates only).

    Mirrors the paper's get_batch (random segment start per trajectory) with
    the boundary restriction that every decision date in the segment belongs
    to the split. RTG values are the logged (realized) return-to-go — the
    conditioning target — so segments near the split end carry RTGs computed
    over the full logged trajectory (see module docstring).

    Policy selection:
    - ``balanced_families=False`` (paper default): a policy is drawn UNIFORMLY
      over all trajectories — the mean_reversion family (20 windows) would
      then supply ~67% of the BC signal and hijack the anchor.
    - ``balanced_families=True``: FIRST a family is drawn uniformly over the
      families present (momentum / mean_reversion / buy_and_hold / random),
      THEN a specific policy/window uniformly inside that family. Each family
      contributes 1/N_families of every batch, restoring a broad BC prior
      regardless of how many windows a family has.

    Returns states/actions/rewards/rtgs/timesteps plus ``policy_idx`` (the
    sampled policy index per batch element, for family-accounting).
    """
    n_pol, t_len = split.actions.shape[:2]
    max_start = t_len - u
    starts = rng.integers(0, max_start + 1, size=batch_size)
    if balanced_families:
        families = sorted({_family_of(p) for p in split.policies})
        fam_policies = {
            f: [i for i, p in enumerate(split.policies) if _family_of(p) == f]
            for f in families
        }
        fam_choice = rng.integers(0, len(families), size=batch_size)
        pols = np.empty(batch_size, dtype=np.int64)
        for j in range(batch_size):
            members = fam_policies[families[fam_choice[j]]]
            pols[j] = members[rng.integers(0, len(members))]
    else:
        pols = rng.integers(0, n_pol, size=batch_size)

    def take(t: torch.Tensor) -> torch.Tensor:
        if t.shape[0] == t_len:  # shared states (T, F), no policy dim
            return torch.stack([t[s : s + u] for s in starts], dim=0)
        return torch.stack(
            [t[pols[i], starts[i] : starts[i] + u] for i in range(batch_size)], dim=0
        )

    return {
        "states": take(split.states),        # (B, u, F)
        "actions": take(split.actions),      # (B, u)
        "rewards": take(split.rewards),      # (B, u) immediate rewards (critic TD)
        "rtgs": take(split.rtgs),            # (B, u) return-to-go (actor conditioning)
        "timesteps": take(split.timesteps),  # (B, u) absolute positions
        "policy_idx": pols,
    }