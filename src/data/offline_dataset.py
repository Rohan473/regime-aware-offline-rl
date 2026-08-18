"""Assemble the offline RL dataset: (s, a, r, s', done) across behavior policies.

Pipeline stages (each cached under data/processed/):
  daily_ohlcv.parquet           minute bars resampled to daily, manifest-guarded
  features_regimes.parquet      OHLCV + raw features + regime column
  offline_dataset.parquet       transitions, tagged with generating policy
  dataset_manifest.json         provenance + summary stats

State = z-scored features at t (8 dims); next_state = same at t+1.
Action = scalar position in [position_min, position_max].
Reward = a_t * ret_{t+1} - cost * |a_t|.
Regime (of t) is carried as a column for later regime-conditional evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .behavior_policies import run_all_policies
from .loaders import REPO_ROOT, load_daily_ohlcv, resolve_path
from .regime_labeling import label_regimes
from .technical_factors import FEATURE_COLUMNS, add_features, normalize_features

STATE_COLUMNS = [f"z_{c}" for c in FEATURE_COLUMNS]


def build_feature_regime_frame(daily: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """daily OHLCV + raw features + regime column (single frame)."""
    features = add_features(daily, cfg)
    if "regime_file" in cfg:
        regime_file = resolve_path(cfg, "regime_file")
    else:
        regime_file = REPO_ROOT / "configs" / "regimes.yaml"
    regime_cfg = OmegaConf.load(str(regime_file))
    features["regime"] = label_regimes(features["close"], regime_cfg)
    return features


def _regime_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts().to_dict()
    return {k: int(v) for k, v in sorted(counts.items())}


def build_offline_dataset(cfg: DictConfig, daily: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: daily OHLCV -> features+regimes -> policy trajectories.

    ``daily`` is injected in tests (synthetic); when None the real loader
    (with its cache + loud failure modes) is used. Never substitutes
    synthetic data for real data in the production path.
    """
    if daily is None:
        daily = load_daily_ohlcv(cfg)

    frame = build_feature_regime_frame(daily, cfg)

    z = normalize_features(frame[FEATURE_COLUMNS], cfg)
    frame = pd.concat([frame, z], axis=1)

    daily_returns = frame["close"].pct_change()
    transitions = run_all_policies(frame, daily_returns, frame["regime"], cfg)

    if transitions.isna().any().any():
        bad = transitions.columns[transitions.isna().any()].tolist()
        raise ValueError(f"offline dataset contains NaN in columns: {bad}")

    for col in STATE_COLUMNS + [f"next_{c}" for c in STATE_COLUMNS]:
        transitions[col] = transitions[col].astype(np.float32)
    transitions["action"] = transitions["action"].astype(np.float32)
    transitions["reward"] = transitions["reward"].astype(np.float32)

    return transitions, frame


def _policy_action_stats(transitions: pd.DataFrame) -> dict:
    stats = {}
    for policy, grp in transitions.groupby("policy"):
        stats[policy] = {
            "n": int(len(grp)),
            "share": round(len(grp) / len(transitions), 4),
            "action_mean": round(float(grp["action"].mean()), 4),
            "action_std": round(float(grp["action"].std()), 4),
            "action_min": round(float(grp["action"].min()), 4),
            "action_max": round(float(grp["action"].max()), 4),
            "reward_mean": round(float(grp["reward"].mean()), 6),
        }
    return stats


def save_dataset(transitions: pd.DataFrame, frame: pd.DataFrame, cfg: DictConfig) -> dict:
    """Persist dataset artifacts + provenance manifest; returns the manifest."""
    processed_dir = resolve_path(cfg, "data.processed_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)

    frame.to_parquet(processed_dir / "features_regimes.parquet")
    transitions.to_parquet(processed_dir / "offline_dataset.parquet")

    manifest = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "date_range": {
            "start": str(transitions["date"].min().date()),
            "end": str(transitions["date"].max().date()),
        },
        "n_transitions": int(len(transitions)),
        "n_days_used": int(transitions["date"].nunique()),
        "regime_counts_days": _regime_counts(
            transitions.drop_duplicates("date")["regime"]
        ),
        "regime_counts_transitions": _regime_counts(transitions["regime"]),
        "policies": _policy_action_stats(transitions),
        "state_dim": len(STATE_COLUMNS),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    (processed_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    return manifest


def summarize(manifest: dict) -> str:
    """Human-readable summary table of the generated dataset."""
    p = manifest["policies"]
    rows = [
        f"{name:<16} {s['n']:>8,} {s['share']*100:>7.2f}% "
        f"{s['action_mean']:>8.4f} {s['action_std']:>8.4f} "
        f"{s['action_min']:>7.3f} {s['action_max']:>7.3f} {s['reward_mean']:>10.6f}"
        for name, s in p.items()
    ]
    regime_rows = [
        f"  {regime:<8} {cnt:>8,} days ({cnt/sum(manifest['regime_counts_days'].values())*100:>5.1f}%)"
        for regime, cnt in manifest["regime_counts_days"].items()
    ]
    return (
        f"OFFLINE DATASET SUMMARY\n"
        f"  date range       : {manifest['date_range']['start']} .. {manifest['date_range']['end']}\n"
        f"  transitions      : {manifest['n_transitions']:,}\n"
        f"  state dim        : {manifest['state_dim']}\n"
        f"  regime breakdown (per unique day):\n" + "\n".join(regime_rows) + "\n"
        f"  policy action stats (mean std min max | reward mean):\n"
        f"  {'policy':<16} {'n':>8} {'share':>8} {'a_mean':>8} {'a_std':>8} "
        f"{'a_min':>7} {'a_max':>7} {'r_mean':>10}\n"
        + "\n".join(rows)
    )
