"""NR7 suite eval: fix_a_mean_nr7 (double-Q mean + uniform + nr7 family).

Reports per the project protocol (PROJECT_NOTES 7.9.1 CHECK 5 / 7.9.2):
per-seed EM MARGINS alongside win counts, never the count alone.

Outputs under src/models/tacr/checkpoints/tacr/<tag>/:
- regime_eval.csv, basin_screening.csv, final_epoch_check.csv,
  margins.csv, vs_baseline.csv
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import regime_eval  # noqa: E402
from src.models.tacr.config import TACRConfig  # noqa: E402
from src.models.tacr.data import load_tacr_data, split_tacr_data  # noqa: E402
from src.models.tacr.eval import (  # noqa: E402
    SEEDS,
    exposure_and_turnover,
    load_checkpoint,
    model_b_and_em,
    roll_actions,
    roll_test_predictions,
)

TAG = "fix_a_mean_nr7"
BASIN_EPOCH_FRAC = 0.35
CORR_FLAG = 0.7


def margins_row(preds: pd.DataFrame, seed: int) -> dict:
    a = preds["action"].to_numpy()
    m = preds["market_ret"].to_numpy()
    s = float(regime_eval.sharpe_ratio(a * m))
    e = float(regime_eval.sharpe_ratio(np.abs(a) * m))
    return {
        "seed": seed,
        "sharpe": round(s, 4),
        "own_EM": round(e, 4),
        "margin": round(s - e, 4),
        "short_frac": round(float((a < 0).mean()), 4),
        "beats_EM": bool(s > e),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=TAG)
    args = parser.parse_args()

    base_cfg = TACRConfig.from_yaml()
    ckpt_root = base_cfg.checkpoint_dir / args.tag
    if not ckpt_root.exists():
        raise FileNotFoundError(f"no checkpoints under {ckpt_root}")
    ckpt_root.mkdir(parents=True, exist_ok=True)

    data = load_tacr_data(base_cfg.u)
    splits = split_tacr_data(data)
    test = splits["test"]

    preds_by_seed, rows, margin_rows, logs = {}, [], [], {}
    for seed in SEEDS:
        c = replace(base_cfg, seed=seed, checkpoint_dir=ckpt_root / f"s{seed}")
        preds = roll_test_predictions(c, checkpoint=c.checkpoint_dir / "tacr_best.pt", data=data)
        preds = preds[preds["valid"]].copy()
        preds_by_seed[seed] = preds
        margin_rows.append(margins_row(preds, seed))

        log = pd.read_csv(c.checkpoint_dir / "training_log.csv")
        best_ep = int(log["val_sharpe"].idxmax()) + 1
        logs[seed] = (best_ep, len(log))

        expo = exposure_and_turnover(preds)
        for regime in ("bull", "bear", "crisis", "all"):
            mask = preds["regime"].to_numpy() == regime if regime != "all" else np.ones(len(preds), bool)
            r = preds["ret"].to_numpy()[mask]
            if len(r) < 2:
                continue
            rec = {"seed": seed, "regime": regime, "best_val_epoch": best_ep,
                   "n_days": int(len(r)),
                   "sharpe": round(regime_eval.sharpe_ratio(r), 4),
                   "cum": round(float(np.prod(1 + r) - 1), 5)}
            rec.update(expo)
            rows.append(rec)

    margins = pd.DataFrame(margin_rows)
    margins.to_csv(ckpt_root / "margins.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(ckpt_root / "regime_eval.csv", index=False)

    print(f"===== {args.tag}: per-seed all-days Sharpe + OWN-EM margins =====")
    print(margins.to_string(index=False))
    wins = int(margins["beats_EM"].sum())
    print(f"wins vs EM: {wins}/5 | mean margin {margins['margin'].mean():+.4f} "
          f"(B naive_new reference: +0.19; pre-nr7 TACR reference: negative)")

    print("\n===== regime breakdown (per-seed sharpe pivot) =====")
    print(table.pivot_table(index="regime", columns="seed", values="sharpe").round(3).to_string())
    print("CRISIS CAVEAT: 15 test days — never a headline.")

    # basin screening
    A = np.vstack([preds_by_seed[s]["action"].to_numpy() for s in SEEDS])
    mean_corr = {s: float(np.nanmean([np.corrcoef(A[i], A[j])[0, 1]
                                      for j in range(5) if j != i]))
                 for i, s in enumerate(SEEDS)}
    flags = {s: (logs[s][0] > round(BASIN_EPOCH_FRAC * logs[s][1])) or (mean_corr[s] < CORR_FLAG)
             for s in SEEDS}
    basin = pd.DataFrame([{"seed": s, "best_val_epoch": logs[s][0], "n_epochs": logs[s][1],
                           "mean_corr_vs_pack": round(mean_corr[s], 4), "flagged": flags[s]}
                          for s in SEEDS])
    basin.to_csv(ckpt_root / "basin_screening.csv", index=False)
    print("\n===== basin screening =====")
    print(basin.to_string(index=False))

    # final-epoch sanity check (standing protocol)
    print("\n===== final-epoch sanity check =====")
    fin_rows = []
    for seed in SEEDS:
        c = replace(base_cfg, seed=seed, checkpoint_dir=ckpt_root / f"s{seed}")
        fin = c.checkpoint_dir / "tacr_final.pt"
        if not fin.exists():
            continue
        fp = roll_test_predictions(c, checkpoint=fin, data=data)
        fp = fp[fp["valid"]]
        b = margins.loc[margins["seed"] == seed, "sharpe"].iloc[0]
        f = float(regime_eval.sharpe_ratio(fp["ret"].to_numpy()))
        fin_rows.append({"seed": seed, "best_val_sharpe": b, "final_sharpe": round(f, 4),
                         "diff": round(f - b, 4)})
        print(f"seed {seed}: best {b:+.4f} vs final {f:+.4f}")
    if fin_rows:
        pd.DataFrame(fin_rows).to_csv(ckpt_root / "final_epoch_check.csv", index=False)

    # vs B / EM baselines
    b_rows = model_b_and_em()
    vs = []
    for policy, group in b_rows.groupby("policy"):
        sh = group["ret"].apply(regime_eval.sharpe_ratio).to_numpy()
        vs.append({"policy": policy, "all_sharpe_mean": round(float(sh.mean()), 4),
                   "all_sharpe_std": round(float(sh.std()), 4)})
    sh = margins["sharpe"].to_numpy()
    vs.append({"policy": f"TACR {args.tag}", "all_sharpe_mean": round(float(sh.mean()), 4),
               "all_sharpe_std": round(float(sh.std()), 4)})
    vs_df = pd.DataFrame(vs)
    vs_df.to_csv(ckpt_root / "vs_baseline.csv", index=False)
    print("\n===== vs B / EM =====")
    print(vs_df.to_string(index=False))
    print(f"\nall outputs under {ckpt_root}")


if __name__ == "__main__":
    main()