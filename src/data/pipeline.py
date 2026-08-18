"""End-to-end data pipeline runner.

Usage (from the repo root):
    python -m src.data.pipeline

Loads configs/data.yaml + configs/regimes.yaml, builds the daily OHLCV
cache, computes features and regime labels, rolls all behavior policies,
and writes the offline dataset + manifest under data/processed/.
"""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

from .loaders import REPO_ROOT
from .offline_dataset import build_offline_dataset, save_dataset, summarize


def validate_config(cfg) -> None:
    problems = []
    f = cfg.features
    if not OmegaConf.is_list(f.returns) or len(f.returns) == 0:
        problems.append("features.returns must be a non-empty list")
    if f.normalize_state and f.normalize_mode not in ("expanding", "rolling"):
        problems.append("features.normalize_mode must be 'expanding' or 'rolling'")
    if int(f.volume_zscore_window) < 2 or int(f.realized_vol_window) < 2:
        problems.append("vol/z-score windows must be >= 2")
    r = cfg.regimes
    if not (0 < float(r.crisis.percentile) < 100):
        problems.append("regimes.crisis.percentile must be in (0, 100)")
    if int(r.hysteresis.crisis_confirmation_days) < 1:
        problems.append("crisis_confirmation_days must be >= 1")
    if int(r.hysteresis.min_regime_duration_days) < 1:
        problems.append("min_regime_duration_days must be >= 1")
    if float(cfg.behavior_policies.position_min) > float(cfg.behavior_policies.position_max):
        problems.append("position_min must be <= position_max")
    for name, p in cfg.behavior_policies.policies.items():
        if name not in ("momentum", "mean_reversion", "buy_and_hold", "random"):
            problems.append(f"unknown behavior policy: {name!r}")
    if problems:
        raise ValueError("config validation failed:\n  - " + "\n  - ".join(problems))


def main() -> None:
    config_dir = REPO_ROOT / "configs"
    cfg = OmegaConf.load(str(config_dir / "data.yaml"))
    cfg.regimes = OmegaConf.load(str(config_dir / "regimes.yaml")).regimes
    cfg.regime_file = str(Path("configs") / "regimes.yaml")
    validate_config(cfg)

    transitions, frame = build_offline_dataset(cfg)
    manifest = save_dataset(transitions, frame, cfg)
    print()
    print(summarize(manifest))
    print(f"\nArtifacts written to {REPO_ROOT / 'data' / 'processed'}")


if __name__ == "__main__":
    sys.exit(main())
