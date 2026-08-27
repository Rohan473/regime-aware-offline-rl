"""Strategy 3 — hybrid: TACR magnitude x linear sign (PROJECT_NOTES 7.17).

The TACR transformer (fix_a_nr7 pack) provides the LEVERAGE TIMING |a_t|;
a simple L2 logistic regression on the same 8 normalized features provides
the DIRECTION sign(r_{t+1}). Composite action:

    a_t = sign_linear(s_t) * |a_TACR(s_t)|

The sign model is trained on the TRAIN split only; the L2 strength C is
selected on the VAL split; the TEST split is untouched for any selection.

Eval (5-seed pack, fix_a_nr7's per-seed checkpoints):
- margin per seed = Sharpe(a_hybrid * m) - Sharpe(|a_TACR| * m)  (the
  hybrid's own exposure-matched control, isolating the sign contribution)
- pre-registered success bar: margin > 0 in >= 3/5 seeds AND pack mean
  test Sharpe > 0.99.

Run: python scripts/hybrid_sign.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.eval.regime_eval import sharpe_ratio  # noqa: E402
from src.models.tacr.config import TACRConfig  # noqa: E402
from src.models.tacr.data import load_tacr_data, split_tacr_data  # noqa: E402
from src.models.tacr.eval import roll_test_predictions  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
TAG = "fix_a_nr7"
C_GRID = [0.01, 0.1, 1.0, 10.0]
CHECKPOINTS = Path(__file__).resolve().parents[1] / "src" / "models" / "tacr" / "checkpoints" / "tacr"


def _signed_targets(market_returns: np.ndarray) -> np.ndarray:
    y = np.sign(market_returns)
    y[y == 0] = 1
    return y


def fit_sign_model(
    train, val
) -> tuple[LogisticRegression, float, float, float]:
    """L2 logistic sign model on train, C selected on val.

    Returns (model, best_C, val_accuracy, always_positive_val_acc)."""
    Xtr = train.states.numpy()
    ytr = _signed_targets(train.market_returns.numpy())
    Xv = val.states.numpy()
    yv = _signed_targets(val.market_returns.numpy())

    baseline = max(float((yv == 1).mean()), float((yv == -1).mean()))
    best_C, best_acc, best_model = None, -1.0, None
    for C in C_GRID:
        m = LogisticRegression(penalty="l2", C=C, max_iter=2000).fit(Xtr, ytr)
        acc = float((m.predict(Xv) == yv).mean())
        if acc > best_acc:
            best_C, best_acc, best_model = C, acc, m
    return best_model, best_C, best_acc, baseline


def eval_hybrid(
    model: LogisticRegression,
    data,
    test,
    seed: int,
    rtg_target: float,
) -> tuple[float, float, float, float]:
    """Roll TACR magnitude for one seed, apply the linear sign, return
    (test_sharpe, margin, short_frac, sign_accuracy_on_test_days_used)."""
    cfg = TACRConfig.from_yaml()
    cfg.rtg_target = rtg_target
    cfg.checkpoint_dir = CHECKPOINTS / TAG / f"s{seed}"
    p = roll_test_predictions(
        cfg, checkpoint=cfg.checkpoint_dir / "tacr_best.pt", data=data
    )
    p = p[p["valid"]].copy()
    pos = test.dates.get_indexer(pd.DatetimeIndex(p["date"]))
    s_test = test.states.numpy()[pos]
    m = p["market_ret"].to_numpy()
    y = _signed_targets(m)
    sign = model.predict(s_test)
    a_tacr = p["action"].to_numpy()
    a_hybrid = sign * np.abs(a_tacr)

    sh_hyb = sharpe_ratio(a_hybrid * m)
    em = sharpe_ratio(np.abs(a_tacr) * m)
    margin = sh_hyb - em
    acc = float((sign == y).mean())
    return sh_hyb, margin, float(np.mean(a_hybrid < 0)), acc


def main() -> None:
    cfg = TACRConfig.from_yaml()
    data = load_tacr_data(20, exclude_policies=cfg.exclude_policies)
    splits = split_tacr_data(data)
    train, val, test = splits["train"], splits["val"], splits["test"]

    model, best_C, val_acc, val_baseline = fit_sign_model(train, val)
    yt = _signed_targets(test.market_returns.numpy())
    test_acc = float((model.predict(test.states.numpy()) == yt).mean())
    test_baseline = max(float((yt == 1).mean()), float((yt == -1).mean()))
    print(f"sign model: C={best_C} | val acc {val_acc:.4f} (always+1 {val_baseline:.4f}) "
          f"| test acc {test_acc:.4f} (always+1 {test_baseline:.4f})")

    for rtg in (0.0, 0.356):
        rows = []
        for seed in SEEDS:
            sh, margin, short, acc = eval_hybrid(model, data, test, seed, rtg)
            rows.append(
                dict(seed=seed, test=round(sh, 4), margin=round(margin, 4),
                     short=round(short, 3), sign_acc=round(acc, 4))
            )
        df = pd.DataFrame(rows)
        wins = int((df.margin > 0).sum())
        mean_sh = float(df.test.mean())
        bar = (wins >= 3) and (mean_sh > 0.99)
        print(f"\n=== hybrid (rtg_target={rtg}) ===")
        print(df.to_string(index=False))
        print(f"margin>0 {wins}/5 | pack mean test {mean_sh:.4f} +- {df.test.std():.4f} "
              f"| SUCCESS BAR (>=3/5 & >0.99): {'PASS' if bar else 'FAIL'}")
    df.to_csv(CHECKPOINTS / "hybrid_sign.csv", index=False)


if __name__ == "__main__":
    main()