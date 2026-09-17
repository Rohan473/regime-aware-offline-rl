"""Decision-layer diagnostics: divergence, action-distribution matching,
regime-conditioned utility, and temporal extrapolation.

For every frozen representation and decision algorithm (BC / A2C / IQL / CQL)
we train the head and report, on the TEST split:

  divergence     mean |a_policy - a_behavior| and corr with the per-date
                 behavior-mean action (how far the learned policy moves from
                 the logged behavior)
  action dist    policy vs behavior action mean/std and Jensen-Shannon
                 divergence of their histograms (distribution matching)
  regime utility test Sharpe within bull / bear / crisis
  temporal       test Sharpe within each calendar year 2021..2024 (forward
                 extrapolation / drift)

usage:
  python scripts/rep_rl_diagnostics.py [--seeds 3] [--reps raw,auto,predictive,contrastive]

Output: data/interpret/rep_rl_diagnostics.csv + printed tables.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders import REPO_ROOT
from src.eval.regime_eval import sharpe_ratio
from src.models.rep_lab.config import OBJECTIVES, RepLabConfig
from src.models.rep_lab.offline_rl import ALGOS, load_offline_rep, train_offline

OUT_DIR = ROOT / "data" / "interpret"
SEEDS = [20260814, 111, 222]
REPS = ["raw", *OBJECTIVES]


def _regime_of(dates: pd.DatetimeIndex) -> np.ndarray:
    feats = pd.read_parquet(REPO_ROOT / "data" / "processed" / "features_regimes.parquet")
    idx = feats.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    reg = pd.Series(feats["regime"].to_numpy(), index=idx)
    d = dates.tz_localize(None) if getattr(dates, "tz", None) is not None else dates
    return reg.reindex(d).to_numpy()


def _js(p: np.ndarray, q: np.ndarray, bins: int = 20) -> float:
    hp, _ = np.histogram(p, bins=bins, range=(-1.0, 1.0))
    hq, _ = np.histogram(q, bins=bins, range=(-1.0, 1.0))
    hp = hp / max(hp.sum(), 1)
    hq = hq / max(hq.sum(), 1)
    return float(jensenshannon(hp, hq))


def _group_sharpe(sr: np.ndarray, groups: np.ndarray) -> dict:
    out = {}
    for g in pd.unique(groups):
        if g is None or (isinstance(g, float) and np.isnan(g)):
            continue
        m = groups == g
        if m.sum() >= 2:
            out[str(g)] = float(sharpe_ratio(sr[m], 252))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default=",".join(REPS))
    ap.add_argument("--algos", default=",".join(ALGOS))
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reps = [r for r in args.reps.split(",") if r]
    algos = [a for a in args.algos.split(",") if a]
    seeds = SEEDS[: args.seeds]

    rows = []

    def add(rep, algo, seed, metric, value, group="all"):
        rows.append({"rep": rep, "algo": algo, "seed": seed, "metric": metric,
                     "group": group, "value": float(value) if np.isfinite(value) else np.nan})

    for rep in reps:
        cfg0 = RepLabConfig()
        data = load_offline_rep(rep, cfg0)
        te = data.split("test")
        dates = data.dates[te]
        regimes = _regime_of(dates)
        years = np.asarray(dates.year)
        r = data.market_returns[te].numpy()
        behavior = data.actions[:, te].numpy()          # (P, T_test)
        beh_mean = behavior.mean(axis=0)
        beh_pool = behavior.reshape(-1)

        for algo in algos:
            for seed in seeds:
                cfg = RepLabConfig()
                cfg.seed = seed
                head, _ = train_offline(rep, algo, cfg, data, epochs=cfg.epochs)
                with __import__("torch").no_grad():
                    a = head.mu(data.H[te]).numpy()
                sr = a * r

                add(rep, algo, seed, "divergence", np.mean(np.abs(a - beh_mean)))
                cc = np.corrcoef(a, beh_mean)[0, 1] if a.std() > 0 else np.nan
                add(rep, algo, seed, "corr_behavior", cc)
                add(rep, algo, seed, "action_std", a.std())
                add(rep, algo, seed, "behavior_action_std", beh_pool.std())
                add(rep, algo, seed, "js_divergence", _js(a, beh_pool))
                add(rep, algo, seed, "sharpe", sharpe_ratio(sr, 252))
                for g, v in _group_sharpe(sr, regimes).items():
                    add(rep, algo, seed, "sharpe_regime", v, g)
                for g, v in _group_sharpe(sr, years).items():
                    add(rep, algo, seed, "sharpe_year", v, g)
        print(f"[done] {rep}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "rep_rl_diagnostics.csv", index=False)

    pd.set_option("display.width", 240)
    print("\n=== behavior divergence (mean over seeds) ===")
    print(df[df["metric"] == "divergence"].pivot_table(
        index="rep", columns="algo", values="value").round(3).to_string())
    print("\n=== action-distribution JS divergence vs behavior ===")
    print(df[df["metric"] == "js_divergence"].pivot_table(
        index="rep", columns="algo", values="value").round(3).to_string())
    print("\n=== regime-conditioned test Sharpe (predictive rep, mean over seeds) ===")
    d = df[(df["metric"] == "sharpe_regime") & (df["rep"] == "predictive")]
    print(d.pivot_table(index="group", columns="algo", values="value").round(3).to_string())
    print("\n=== per-year test Sharpe (predictive rep, mean over seeds) ===")
    d = df[(df["metric"] == "sharpe_year") & (df["rep"] == "predictive")]
    print(d.pivot_table(index="group", columns="algo", values="value").round(3).to_string())
    print("\nwrote rep_rl_diagnostics.csv under", OUT_DIR)


if __name__ == "__main__":
    main()
