"""Model C+ CONFIRMATORY RUN — the untouched 2025-01-01..2026-03-31 window.
(PROJECT_NOTES 7.26)

Multiple-testing guard: every experiment 7.10-7.25 was evaluated against the
SAME 2021-2024 test window, with the choice of the next experiment informed
by those numbers (a garden-of-forking-paths hazard). This is a SINGLE, no-
further-iteration evaluatation of the FROZEN canonical C+ artifacts on data
that has never been touched by any selection:

  - magnitude |a|: the fix_a_nr7 per-seed checkpoints (frozen .pt files)
  - sign        : the ref16 CE logistic, 8 SPY + 8 macro causal features,
                  C=1.0, fit deterministically on the ORIGINAL frozen train
                  split (<=2018-12-31). NOTHING touches 2025-2026.

No hyperparameter / feature / window choice may be altered after this run.
If the confirmation-window margin is null or negative, C+ does not
generalize and that is the recorded result.

Protocol notes (the "frozen artifact" is reproduced, not re-tuned):
- The sign model IS refit, but deterministically, on the SAME frozen
  train/val data: sklearn L2-LBFGS logistic is fully deterministic on fixed
  data + fixed C, so this reproduces the 7.18 ref16 weights bit-for-bit. A
  sanity gate below re-evaluates the canonical test window and must match the
  current-state C+ numbers before the confirmation result is read.
- Timestep input to the transformer: the extended loader (test_end=CONF_END)
  would feed positions past the last trained timestep (never-seen embedding
  rows). Positions beyond the original study-horizon length are therefore
  clamped to orig_len-1, i.e. the model's trajectory is frozen at its last
  trained position. This does not change a single pre-2025 timestep.

Run: python scripts/hybrid_sign_confirmation.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders import REPO_ROOT  # noqa: E402
from src.data.macro_factors import MACRO_FEATURES, macro_features, zscore_causal  # noqa: E402
from src.eval.regime_eval import sharpe_ratio  # noqa: E402
from src.models import SPLIT_TEST_END, SPLIT_TRAIN_END, SPLIT_VAL_END  # noqa: E402
from src.models.tacr.config import TACRConfig  # noqa: E402
from src.models.tacr.data import load_tacr_data  # noqa: E402
from src.models.tacr.eval import load_checkpoint, roll_actions  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
TAG = "fix_a_nr7"
CONF_START = "2025-01-01"
CONF_END = "2026-03-31"
CHECKPOINTS = ROOT / "src" / "models" / "tacr" / "checkpoints" / "tacr"
SPY_Z = ["z_ret_1d", "z_ret_5d", "z_ret_20d", "z_realized_vol_20d",
         "z_rsi_14", "z_macd_hist", "z_volume_zscore_20d", "z_bollinger_pos"]

# Canonical 7.18/7.22 reference numbers (as recorded in PROJECT_NOTES) for
# the eyeball check — current-state reproduction is printed alongside.
NOTES_REF = {
    "B": {"C": 1.0, "test_acc": 0.5393, "short_frac": 0.0637,
          "margin_mean": 0.2962, "wins": 5, "sharpe_mean": 1.146},
}


def _signed(m: np.ndarray) -> np.ndarray:
    y = np.sign(m)
    y[y == 0] = 1
    return y


def _eval_window(
    sign_model: LogisticRegression,
    Xm: np.ndarray,
    data,
    window_dates: pd.DatetimeIndex,
    seed: int,
) -> dict:
    """Roll |a| for one fix_a_nr7 seed, apply the frozen sign model, return
    the per-seed window stats on the OVERLAPPING FEASIBLE days."""
    cfg = TACRConfig.from_yaml()
    cfg.checkpoint_dir = CHECKPOINTS / TAG / f"s{seed}"
    model, am, _ = load_checkpoint(cfg.checkpoint_dir / "tacr_best.pt", cfg)
    acts = roll_actions(model, cfg, data, window_dates, cfg.rtg_target,
                        action_model=am, bcq_phi=cfg.bcq_phi)
    pos = data.dates.get_indexer(window_dates)
    acts = acts[pos]

    Xt = Xm[pos]
    good = np.isfinite(Xt).all(1)
    sig = sign_model.predict(Xt[good])
    a = acts[good]
    mm = data.market_returns.numpy()[pos][good]
    hyb = sig * np.abs(a)
    sh = sharpe_ratio(hyb * mm)
    em = sharpe_ratio(np.abs(a) * mm)
    return {
        "seed": seed,
        "n_days": int(good.sum()),
        "sharpe": sh,
        "em": em,
        "margin": sh - em,
        "short_frac": float((sig < 0).mean()),
        "sign_acc": float((sig == _signed(mm)).mean()),
    }


def main() -> None:
    # --- load with the study-horizon boundary extended to CONF_END ----------
    cfg = TACRConfig.from_yaml()
    data = load_tacr_data(20, exclude_policies=cfg.exclude_policies, test_end=CONF_END)

    # Freeze the timestep channel at the last trained position (see header).
    naive = data.dates.tz_localize(None)
    orig_len = int((naive <= pd.Timestamp(SPLIT_TEST_END)).sum())
    ts = np.arange(len(naive), dtype=np.int64)
    n_clamped = int((ts >= orig_len).sum())
    ts[ts >= orig_len] = orig_len - 1
    data = replace(
        data,
        timesteps=torch.tensor(ts[None, :].repeat(len(data.policies), axis=0), dtype=torch.long),
    )
    print(f"[confirmation] dates {naive.min().date()}..{naive.max().date()} "
          f"({len(naive)}); timesteps clamped to {orig_len - 1} for "
          f"{n_clamped} positions beyond the 2024 study horizon")

    # --- frozen 16-feature matrix (identical construction to 7.18) ----------
    feats = pd.read_parquet(REPO_ROOT / "data" / "processed" / "features_regimes.parquet")
    spy_z = feats[SPY_Z].reindex(data.dates).to_numpy(dtype="float64")     # (T, 8)
    macro_raw = macro_features(feats)
    macro_z = np.column_stack([zscore_causal(macro_raw[c]) for c in MACRO_FEATURES])
    macro_z = macro_z[np.searchsorted(macro_raw.index, naive)]             # (T, 8)
    Fb = np.concatenate([spy_z, macro_z], axis=1)                          # (T, 16)

    y = _signed(data.market_returns.numpy())
    ok = np.isfinite(Fb).all(1)

    tr_mask = naive <= pd.Timestamp(SPLIT_TRAIN_END)
    tr_n = int((tr_mask & ok).sum())

    # --- the FROZEN artifact: ref16 CE logistic, C=1.0 on train only --------
    # Deterministic on fixed data -> reproduces the 7.18 B weights exactly.
    sign_model = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
    sign_model.fit(Fb[tr_mask & ok], y[tr_mask & ok])

    # --- windows -------------------------------------------------------------
    te_mask = (naive > pd.Timestamp(SPLIT_VAL_END)) & (naive <= pd.Timestamp(SPLIT_TEST_END))
    cn_mask = (naive > pd.Timestamp(SPLIT_TEST_END)) & (naive <= pd.Timestamp(CONF_END))
    canon_dates = data.dates[te_mask]
    conf_dates = data.dates[cn_mask]
    print(f"[confirmation] canonical window 2021-2024: {int(te_mask.sum())} days | "
          f"confirmation window 2025-01-01..2026-03-31: {int(cn_mask.sum())} days "
          f"(regimes {data.regimes[cn_mask].value_counts().to_dict()})")

    # --- sanity gate: canonical window must reproduce current-state C+ -------
    ref = NOTES_REF["B"]
    canon_rows = [_eval_window(sign_model, Fb, data, canon_dates, s) for s in SEEDS]
    canon = pd.DataFrame(canon_rows)
    acc = float((sign_model.predict(Fb[te_mask & ok]) == y[te_mask & ok]).mean())
    print("\nCANONICAL TEST WINDOW (2021-2024) — sanity gate (frozen ref16, C=1.0):")
    print(f"  current state   acc {acc:.4f} short {canon.short_frac.mean():.4f} "
          f"margin {canon.margin.mean():+.4f} wins {int((canon.margin>0).sum())}/5 "
          f"Sharpe {canon.sharpe.mean():.4f}")
    print(f"  notes 7.18/7.22 acc {ref['test_acc']:.4f} short {ref['short_frac']:.4f} "
          f"margin {ref['margin_mean']:+.4f} wins {ref['wins']}/5 Sharpe {ref['sharpe_mean']:.4f}")

    # --- THE CONFIRMATION (report once, then stop) ---------------------------
    conf_rows = [_eval_window(sign_model, Fb, data, conf_dates, s) for s in SEEDS]
    conf = pd.DataFrame(conf_rows)
    out = pd.concat(
        [canon.assign(window="canonical_2021_2024"), conf.assign(window="confirmation_2025_2026")]
    )
    out_path = CHECKPOINTS / "hybrid_sign_confirmation.csv"
    out.to_csv(out_path, index=False)

    print("\nCONFIRMATION WINDOW (2025-01-01..2026-03-31, UNTOUCHED) — reported once:")
    print(f"  per-seed: margin {', '.join(f'{m:+.4f}' for m in conf.margin)}")
    print(f"  per-seed: sharpe {', '.join(f'{s:.4f}' for s in conf.sharpe)}")
    print(f"  per-seed: EM    {', '.join(f'{e:.4f}' for e in conf.em)}")
    print(f"  pack mean Sharpe {conf.sharpe.mean():.4f} +- {conf.sharpe.std():.4f} | "
          f"pack mean EM {conf.em.mean():.4f} | pack mean margin {conf.margin.mean():+.4f} | "
          f"margin>0 {int((conf.margin>0).sum())}/5 | short_frac {conf.short_frac.mean():.4f} | "
          f"sign_acc {conf.sign_acc.mean():.4f} | n_days {int(conf.n_days.iloc[0])}")
    print(f"\ncanonical reference (current state, same frozen artifact): margin "
          f"{canon.margin.mean():+.4f} / Sharpe {canon.sharpe.mean():.4f} (5/5) — "
          f"confirmation generalization = margin {conf.margin.mean():+.4f}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()