"""Evaluation entry point for Model B (DDR).

Rolls the trained (deterministic) policy over the TEST split, computes the
regime breakdown via the shared ``src/eval/regime_eval`` module (primary
result), and reports blended metrics as secondary. Regime labels are joined
by date from Phase 1 ``features_regimes.parquet`` — never recomputed.

Outputs under ``cfg.checkpoint_dir``:
- ``test_predictions.csv``  date, action, market_ret, ret, cumret, regime
- ``regime_eval.csv``       the regime-breakdown table
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.loaders import REPO_ROOT
from src.eval import regime_eval
from src.models.ddr.config import DDRConfig
from src.models.ddr.data import load_ddr_data, split_ddr_data
from src.models.ddr.policy import DDRPolicy
from src.models.ddr.train import BEST_NAME, _strategy_returns


def roll_test_predictions(
    cfg: DDRConfig, checkpoint: Path | None = None, data=None
) -> pd.DataFrame:
    """Deterministic test-set roll; returns date, action, ret, cumret, regime."""
    if checkpoint is None:
        checkpoint = cfg.checkpoint_dir / BEST_NAME
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"no checkpoint at {checkpoint} — run `python -m src.models.ddr.train` first"
        )
    payload = torch.load(checkpoint, map_location="cpu")
    model = DDRPolicy(input_dim=8, hidden_size=cfg.hidden_size, rnn_type=cfg.rnn_type)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    if data is None:
        data = load_ddr_data(cfg.window_size)
    test = split_ddr_data(data)["test"]
    if len(test.windows) == 0:
        raise ValueError("test split is empty")

    with torch.no_grad():
        actions = model(test.windows).squeeze(-1)
    returns = _strategy_returns(actions, test.next_returns, cfg.transaction_cost_bps)
    dates = pd.DatetimeIndex(test.dates)
    preds = pd.DataFrame(
        {
            "date": dates,
            "action": actions.numpy(),
            "market_ret": test.next_returns.numpy(),
            "ret": returns.numpy(),
            "cumret": np.cumprod(1.0 + returns.numpy()) - 1.0,
        }
    )
    preds["regime"] = test.regimes.to_numpy()
    valid = (
        test.valid.numpy()
        if test.valid is not None
        else np.ones(len(test.windows), dtype=bool)
    )
    preds["valid"] = valid
    return preds


def baseline_policy_table(cfg: DDRConfig, data=None) -> pd.DataFrame:
    """Regime breakdown of the four Phase-1 behavior policies on the TEST
    split — the reference point DDR must beat.

    Strategy return per day = logged action * realized next-day market return
    (transaction cost 0, matching Phase-1 reward with cost_bps=0). Valid-day
    filter applies (data-hole boundaries excluded).
    """
    if data is None:
        data = load_ddr_data(cfg.window_size)
    test = split_ddr_data(data)["test"]
    valid = (
        test.valid.numpy()
        if test.valid is not None
        else np.ones(len(test.windows), dtype=bool)
    )
    test_dates = test.dates[valid]
    ds = pd.read_parquet(REPO_ROOT / "data" / "processed" / "offline_dataset.parquet")
    ds = ds[ds["date"].isin(test_dates)]
    market = pd.Series(test.next_returns.numpy()[valid], index=test_dates)
    rows = []
    for policy, grp in ds.groupby("policy"):
        g = grp.set_index("date").sort_index()
        ret = (g["action"].astype("float64") * market.reindex(g.index)).to_numpy()
        preds = pd.DataFrame(
            {"date": g.index, "ret": ret, "regime": g["regime"].to_numpy()}
        )
        table = regime_eval.evaluate_regime_breakdown(preds)
        rec = {"policy": policy}
        for regime in table.index:
            rec[f"{regime}_sharpe"] = table.loc[regime, "sharpe_annualized"]
            rec[f"{regime}_cum"] = table.loc[regime, "cum_return"]
            rec[f"{regime}_n"] = table.loc[regime, "n_days"]
        rows.append(rec)
    return pd.DataFrame(rows).set_index("policy")


def hole_spanning_flags(features_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Bool per row: does the trailing 20d (features) or 60d (regime-label)
    window contain a data hole? Legit W trading days span ~1.4-1.5*W calendar
    days; holes add 33+ days over that. Threshold 1.6*W + 8."""
    idx = features_index
    flags = pd.DataFrame(index=idx)
    for back, name in ((19, "win20_spans_hole"), (59, "win60_spans_hole")):
        cal = (idx[back:] - idx[:-back]).days
        f = pd.Series(False, index=idx)
        f.iloc[back:] = cal.to_numpy() > 1.6 * (back + 1) + 8
        flags[name] = f
    return flags


def clean_label_mask(test_dates: pd.DatetimeIndex) -> np.ndarray:
    """True for test days whose 60d regime-label window contains no data
    hole. These are the ONLY days whose regime label is trustworthy; the
    remaining days' labels are computed on hole-spanning windows and must
    not be reported as regime evidence."""
    features = pd.read_parquet(REPO_ROOT / "data" / "processed" / "features_regimes.parquet")
    return (~hole_spanning_flags(features.index)["win60_spans_hole"]).reindex(test_dates).to_numpy()


def evaluate(cfg: DDRConfig, checkpoint: Path | None = None, data=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = roll_test_predictions(cfg, checkpoint=checkpoint, data=data)
    # data-hole boundary days are not daily returns; exclude from metrics
    # (rows are kept in the CSV with their `valid` flag)
    table = regime_eval.evaluate_regime_breakdown(preds[preds["valid"]])
    n_excluded = int((~preds["valid"]).sum())
    if n_excluded:
        print(f"[eval] excluded {n_excluded} data-hole boundary day(s) from metrics")
    n_label_contaminated = int((preds["valid"] & ~clean_label_mask(preds["date"])).sum())
    if n_label_contaminated:
        print(f"[eval] WARNING: {n_label_contaminated} more valid days have regime LABELS")
        print("[eval] computed on hole-spanning windows; the table below mixes them.")
        print("[eval] Trusted regime evidence is the clean-label set: "
              f"{int(preds['valid'].sum()) - n_label_contaminated} days.")
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(cfg.checkpoint_dir / "test_predictions.csv", index=False)
    table.to_csv(cfg.checkpoint_dir / "regime_eval.csv")
    return preds, table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Model B (DDR) on the test set.")
    parser.add_argument("--config", type=str, default=None, help="ddr.yaml path (default configs/ddr.yaml)")
    parser.add_argument("--checkpoint", type=str, default=None, help="override checkpoint path")
    args = parser.parse_args(argv)
    cfg = DDRConfig.from_yaml(args.config) if args.config else DDRConfig.from_yaml()
    preds, table = evaluate(cfg, checkpoint=Path(args.checkpoint) if args.checkpoint else None)
    print(table.to_string())
    print(f"\nheadline (test {preds['date'].min().date()} .. {preds['date'].max().date()}, "
          f"{len(preds)} days): see regime_eval.csv; blended = secondary")
    print(f"predictions: {cfg.checkpoint_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()