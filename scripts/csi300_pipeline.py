"""CSI300 data pipeline: generate features, regimes, behavior policies, offline dataset.

Reads the downloaded CSI300 daily CSV and produces the same artifacts the
models expect under data/processed/:
  - daily_ohlcv.parquet
  - features_regimes.parquet
  - offline_dataset.parquet
  - dataset_manifest.json

Usage:
    python scripts/csi300_pipeline.py [--cost-bps 1.0]

--cost-bps: turnover cost (bps) charged on the offline-dataset reward
(regenerated at the cost level specified, default 0.0). Features/regimes
are cost-independent and byte-identical across runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_csi300_daily(csv_path: Path) -> pd.DataFrame:
    """Load CSI300 daily CSV into the format the pipeline expects."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.index.name = "date"
    # Ensure columns match what the pipeline expects
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"missing column: {col}")
    # Drop rows with zero volume (pre-launch back-calculated data)
    before = len(df)
    df = df[df["volume"] > 0].copy()
    if len(df) < before:
        print(f"[csi300] dropped {before - len(df)} rows with volume=0 (pre-launch back-calc)")
    print(f"[csi300] loaded {len(df)} trading days: {df.index.min().date()} to {df.index.max().date()}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="CSI300 data pipeline.")
    parser.add_argument(
        "--cost-bps", type=float, default=0.0,
        help="turnover cost (bps) charged on the offline-dataset reward",
    )
    args = parser.parse_args()

    from src.data.technical_factors import add_features, normalize_features, FEATURE_COLUMNS
    from src.data.regime_labeling import label_regimes
    from src.data.behavior_policies import run_all_policies
    from src.data.offline_dataset import summarize

    csv_path = REPO_ROOT / "data" / "csi300_daily.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run download first.", file=sys.stderr)
        sys.exit(1)

    # Load CSI300 daily data
    daily = load_csi300_daily(csv_path)

    # Load configs (use the same features/regimes/behavior_policies as SPY)
    config_dir = REPO_ROOT / "configs"
    cfg = OmegaConf.load(str(config_dir / "data.yaml"))
    cfg.regimes = OmegaConf.load(str(config_dir / "regimes.yaml")).regimes
    cfg.regime_file = str(Path("configs") / "regimes.yaml")
    cfg.dataset.transaction_cost_bps = float(args.cost_bps)
    print(f"\n[csi300] offline reward turnover cost: {args.cost_bps:.1f} bps")

    # Override data source to skip minute-bar loading
    # We pass daily directly to build_offline_dataset
    print("\n[csi300] computing features...")
    features = add_features(daily, cfg)
    print(f"[csi300] features shape: {features.shape}")

    print("[csi300] labeling regimes...")
    regime_cfg = OmegaConf.load(str(config_dir / "regimes.yaml"))
    features["regime"] = label_regimes(features["close"], regime_cfg)
    regime_counts = features["regime"].value_counts()
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} days")

    print("[csi300] normalizing features...")
    z = normalize_features(features[FEATURE_COLUMNS], cfg)
    features = pd.concat([features, z], axis=1)

    print("[csi300] rolling behavior policies...")
    daily_returns = features["close"].pct_change()
    transitions = run_all_policies(features, daily_returns, features["regime"], cfg)
    print(f"[csi300] transitions: {len(transitions)} rows, {transitions['policy'].nunique()} policies")

    if transitions.isna().any().any():
        bad = transitions.columns[transitions.isna().any()].tolist()
        raise ValueError(f"offline dataset contains NaN in columns: {bad}")

    import numpy as np
    state_columns = [f"z_{c}" for c in FEATURE_COLUMNS]
    for col in state_columns + [f"next_{c}" for c in state_columns]:
        transitions[col] = transitions[col].astype(np.float32)
    transitions["action"] = transitions["action"].astype(np.float32)
    transitions["reward"] = transitions["reward"].astype(np.float32)

    # Save artifacts
    processed_dir = REPO_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    features.to_parquet(processed_dir / "features_regimes.parquet")
    transitions.to_parquet(processed_dir / "offline_dataset.parquet")

    # Build manifest
    def _regime_counts(series):
        counts = series.value_counts().to_dict()
        return {k: int(v) for k, v in sorted(counts.items())}

    def _policy_action_stats(df):
        stats = {}
        for policy, grp in df.groupby("policy"):
            stats[policy] = {
                "n": int(len(grp)),
                "share": round(len(grp) / len(df), 4),
                "action_mean": round(float(grp["action"].mean()), 4),
                "action_std": round(float(grp["action"].std()), 4),
                "action_min": round(float(grp["action"].min()), 4),
                "action_max": round(float(grp["action"].max()), 4),
                "reward_mean": round(float(grp["reward"].mean()), 6),
            }
        return stats

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
        "state_dim": len(state_columns),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    (processed_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print()
    print(summarize(manifest))
    print(f"\nArtifacts written to {processed_dir}")


if __name__ == "__main__":
    sys.exit(main())
