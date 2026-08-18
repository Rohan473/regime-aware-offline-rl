"""Volatility-targeted DSR retrain + leverage-matched control (reward fix).

The naive DSR reward (Moody-Saffell differential Sharpe) lets the policy
shrink |a_t| to reduce the reward's variance -- that is exactly what the
deployed checkpoints do (mean |a| ~0.3-0.4, shorts ~5-15%), and it is why
DDR lost to the exposure-matched control (EM = |a_t| always long) 5/5
seeds. Vol-targeting fixes the reward: every action is scaled so the
strategy-return series runs at target_vol annualized BEFORE the DSR
increment, so de-leveraging no longer lowers the reward's variance (the
scaled series is invariant to uniform |a| shrinkage).

This script:
  1. trains 5 seeds with vol-targeting (checkpoints/vt/s{seed}/ -- the OLD
     naive checkpoints in checkpoints/ and checkpoints/robustness/ are left
     untouched),
  2. reruns the leverage decomposition (B&H / CL / EM / DDR) on clean-label
     test days for BOTH the vol-targeted and the naive checkpoints,
  3. applies the verdict rule: vol-targeted DDR must beat EM in >= 3/5
     seeds. PASS  -> the reward was the problem; the signal exists.
     FAIL   -> GRU+DSR cannot extract signal from this feature set at this
               window; report that explicitly and STOP iterating on reward
               shape (the next lever is the feature set, not the reward).

Outputs: checkpoints/vt/leverage_test.csv, verdict printed to stdout.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import regime_eval  # noqa: E402
from src.models.ddr.config import DDRConfig  # noqa: E402
from src.models.ddr.data import load_ddr_data, split_ddr_data  # noqa: E402
from src.models.ddr.eval import roll_test_predictions  # noqa: E402
from src.models.ddr.train import train_ddr  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
VT_DIR = DDRConfig().checkpoint_dir / "vt"          # checkpoints/vt (vol-targeted, CURRENT data)
NAIVE_NEW_DIR = DDRConfig().checkpoint_dir / "naive_new"  # naive DSR retrained on CURRENT data
NAIVE_OLD_DIR = DDRConfig().checkpoint_dir / "robustness"  # naive per-seed runs (OLD contaminated data, historical)


def clean_masks(data) -> np.ndarray:
    features = pd.read_parquet(ROOT / "data" / "processed" / "features_regimes.parquet")
    idx = features.index
    cal = (idx[59:] - idx[:-59]).days
    f = pd.Series(False, index=idx)
    f.iloc[59:] = cal.to_numpy() > 1.6 * 60 + 8
    test = split_ddr_data(data)["test"]
    valid = test.valid.numpy()
    return (~f.reindex(test.dates[valid])).to_numpy()


def train_seeds(cfg: DDRConfig, out_dir: Path) -> None:
    for seed in SEEDS:
        c = replace(cfg, seed=seed, checkpoint_dir=out_dir / f"s{seed}")
        ckpt = c.checkpoint_dir / "ddr_best.pt"
        if ckpt.exists():
            print(f"[{out_dir.name}] seed {seed}: reusing {ckpt}")
            continue
        model, history, checkpoint = train_ddr(c)
        print(f"[{out_dir.name}] seed {seed}: best val Sharpe {history['val_sharpe_ann'].max():.3f} "
              f"@ epoch {int(history['val_sharpe_ann'].idxmax())}")


def per_seed_preds(cfg: DDRConfig, data) -> list[pd.DataFrame]:
    out = []
    for seed in SEEDS:
        c = replace(cfg, seed=seed, checkpoint_dir=Path(cfg.checkpoint_dir) / f"s{seed}")
        preds = roll_test_predictions(c, checkpoint=c.checkpoint_dir / "ddr_best.pt", data=data)
        out.append(preds[preds["valid"]].copy())
    return out


def leverage_table(preds_by_seed, clean: np.ndarray) -> pd.DataFrame:
    rows = []
    for seed, preds in zip(SEEDS, preds_by_seed):
        p = preds[clean].copy()
        a = p["action"].to_numpy()
        m = p["market_ret"].to_numpy()
        lam = np.abs(a).mean()
        p["ret_bh"] = m
        p["ret_cl"] = lam * m
        p["ret_em"] = np.abs(a) * m
        p["ret_ddr"] = p["ret"].to_numpy()
        for regime in ("bull", "all"):
            mask = p["regime"].to_numpy() == regime if regime != "all" else np.ones(len(p), bool)
            for name in ("ret_bh", "ret_cl", "ret_em", "ret_ddr"):
                r = p[name].to_numpy()[mask]
                rows.append(
                    {
                        "seed": seed,
                        "regime": regime,
                        "policy": name.replace("ret_", ""),
                        "n_days": int(mask.sum()),
                        "sharpe": round(regime_eval.sharpe_ratio(r), 4),
                        "cum": round(float(np.prod(1 + r) - 1), 6),
                        "max_dd": round(regime_eval.max_drawdown(r), 5),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    data = load_ddr_data(DDRConfig().window_size)
    clean = clean_masks(data)

    print("=== phase 1: retrain on CURRENT (hole-free) data ===")
    print("-- vol-targeted DSR (the fix) --")
    train_seeds(DDRConfig.from_yaml(), VT_DIR)
    print("-- naive DSR (control, same data) --")
    train_seeds(DDRConfig(), NAIVE_NEW_DIR)

    print("\n=== phase 2: leverage decomposition (valid test days) ===")
    vt_cfg = DDRConfig.from_yaml()
    vt = leverage_table(per_seed_preds(vt_cfg, data), clean)
    vt["run"] = "voltarget"
    naive_new = leverage_table(per_seed_preds(replace(DDRConfig(), checkpoint_dir=NAIVE_NEW_DIR), data), clean)
    naive_new["run"] = "naive_new"
    naive_old = leverage_table(per_seed_preds(replace(DDRConfig(), checkpoint_dir=NAIVE_OLD_DIR), data), clean)
    naive_old["run"] = "naive_old"
    table = pd.concat([naive_old, naive_new, vt], ignore_index=True)
    table.to_csv(VT_DIR / "leverage_test.csv", index=False)

    piv = table.pivot_table(index=["run", "regime", "policy"], columns="seed", values="sharpe")
    piv["mean"] = piv.mean(axis=1)
    piv["std"] = piv.std(axis=1)
    print(piv.round(4).to_string())

    def wins(run: str, regime: str) -> int:
        t = table[(table["run"] == run) & (table["regime"] == regime)]
        d = t[t["policy"] == "ddr"].set_index("seed")["sharpe"]
        e = t[t["policy"] == "em"].set_index("seed")["sharpe"]
        return int((d > e).sum())

    print("\n=== VERDICT (criterion: >= 3/5 seeds) ===")
    w_vt = wins("voltarget", "all")
    w_na = wins("naive_new", "all")
    w_vt_b = wins("voltarget", "bull")
    w_na_b = wins("naive_new", "bull")
    print(f"vol-targeted DDR vs EM:  {w_vt}/5 on 'all' days, {w_vt_b}/5 on bull")
    print(f"naive (same data) vs EM: {w_na}/5 on 'all' days, {w_na_b}/5 on bull")
    if w_vt >= 3:
        print("PASS (criterion): with vol-targeting the GRU beats the exposure-matched control.")
    else:
        print("FAIL (criterion): vol-targeting did not rescue the signal -- report the signal")
        print("      problem explicitly and stop iterating on reward shape.")
    if w_na >= 3:
        print("NOTE: naive DSR (same data) ALSO beats EM -- the signal exists regardless of reward")
        print("      shape; the old 'no directional signal' verdict was an artifact of the")
        print("      contaminated (hole-filled) dataset it was measured on.")
    vt_all = table[(table["run"] == "voltarget") & (table["regime"] == "all") & (table["policy"] == "ddr")]
    na_all = table[(table["run"] == "naive_new") & (table["regime"] == "all") & (table["policy"] == "ddr")]
    diff = vt_all.set_index("seed")["sharpe"] - na_all.set_index("seed")["sharpe"]
    print(f"vol-targeted minus naive (same data, 'all'): {diff.mean():+.4f} per seed "
          f"({int((diff > 0).sum())}/5 seeds positive) -- the fix's marginal value on this data")

    print("\nmax drawdown comparison (mean over seeds, 'all' days):")
    dd = table[(table["regime"] == "all") & (table["policy"].isin(["bh", "em", "ddr"]))]\
        .pivot_table(index=["run", "policy"], values="max_dd")
    print(dd.round(5).to_string())


if __name__ == "__main__":
    main()