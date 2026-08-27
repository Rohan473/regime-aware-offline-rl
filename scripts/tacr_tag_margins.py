"""Fast margin-only eval for arbitrary TACR experiment tags.

Computes per-seed all-days Sharpe, OWN-EM Sharpe, margin, short_frac and
the final-epoch check for each requested tag (5-seed packs), writing
<tag>/margins.csv under checkpoints/tacr/. The roll is identical whether
the dataset has 31 or 32 policies (shared states; nr7's date coverage is a
superset of the intersection), so pre-NR7 checkpoints roll correctly on
the current dataset.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.regime_eval import sharpe_ratio  # noqa: E402
from src.models.tacr.config import TACRConfig  # noqa: E402
from src.models.tacr.data import load_tacr_data  # noqa: E402
from src.models.tacr.eval import SEEDS, roll_test_predictions  # noqa: E402

TAGS = sys.argv[1:] or [
    "ctl", "ctl_bal", "ctl_mix", "ctl_nr",
    "fix_a", "fix_a_bal", "fix_a_mean", "fix_a_nr", "fix_b",
]


def main() -> None:
    base_cfg = TACRConfig.from_yaml()
    data = load_tacr_data(base_cfg.u)

    for tag in TAGS:
        root = base_cfg.checkpoint_dir / tag
        if not root.exists():
            print(f"[{tag}] no checkpoint dir, skipped")
            continue
        rows = []
        for seed in SEEDS:
            c = replace(base_cfg, seed=seed, checkpoint_dir=root / f"s{seed}")
            try:
                preds = roll_test_predictions(c, checkpoint=c.checkpoint_dir / "tacr_best.pt", data=data)
            except FileNotFoundError:
                print(f"[{tag}] seed {seed} missing, skipped")
                rows = []
                break
            p = preds[preds["valid"]]
            a, m = p["action"].to_numpy(), p["market_ret"].to_numpy()
            s, e = float(sharpe_ratio(a * m)), float(sharpe_ratio(np.abs(a) * m))
            row = {"seed": seed, "sharpe": round(s, 4), "own_EM": round(e, 4),
                   "margin": round(s - e, 4), "short_frac": round(float((a < 0).mean()), 4),
                   "beats_EM": bool(s > e)}
            fin = c.checkpoint_dir / "tacr_final.pt"
            if fin.exists():
                fp = roll_test_predictions(c, checkpoint=fin, data=data)
                fp = fp[fp["valid"]]
                row["final_sharpe"] = round(float(sharpe_ratio(fp["ret"].to_numpy())), 4)
            rows.append(row)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.to_csv(root / "margins.csv", index=False)
        wins = int(df["beats_EM"].sum())
        print(f"[{tag}] wins {wins}/5 | mean margin {df['margin'].mean():+.4f} | "
              f"mean sharpe {df['sharpe'].mean():+.4f} +- {df['sharpe'].std():.4f} | "
              f"short_frac {df['short_frac'].mean():.3f}"
              + (f" | final-check: {int((df['final_sharpe'] > 0).sum())}/5 positive"
                 if "final_sharpe" in df else ""))
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()