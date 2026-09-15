"""Compile the CSI300 cross-model comparison (Sham/EM table) and save to
data/csi300_model_comparison.csv.

For each model we roll test predictions across the 5-seed pack and report:
  sharpe_model  = Sharpe(action * market_ret)          (the model itself)
  sharpe_em     = Sharpe(|action| * market_ret)        (its OWN EM control)
  margin        = sharpe_model - sharpe_em
  wins_vs_em    = # seeds where model > its own EM  (>=3/5 clears)

Models: B (DDR naive), B-vt (DDR vol-targeted), C (TACR), D, D-minus-fuzzy,
plus the C+ hybrid signs (z-only and z+macro).

Checkpoint roots are overridable via env vars so at-cost retrains can be
compiled without touching the zero-cost packs:
  CSI300_DDR_CKPT  default src/models/ddr/checkpoints
  CSI300_TACR_CKPT default src/models/tacr/checkpoints/tacr   (yaml value)
  CSI300_D_CKPT    default src/models/d/checkpoints
  CSI300_OUT       default data/csi300_model_comparison.csv
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.regime_eval import sharpe_ratio  # noqa: E402
from src.models.ddr.config import DDRConfig  # noqa: E402
from src.models.ddr.data import load_ddr_data  # noqa: E402
from src.models.ddr.eval import roll_test_predictions as ddr_roll  # noqa: E402
from src.models.tacr.config import TACRConfig  # noqa: E402
from src.models.tacr.data import load_tacr_data, split_tacr_data  # noqa: E402
from src.models.tacr.eval import roll_test_predictions as tacr_roll  # noqa: E402
from src.models.d.config import DConfig  # noqa: E402
from src.models.d.data import load_d_data, split_d_data  # noqa: E402
from src.models.d.eval import roll_test_preds  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
OUT = Path(os.environ.get("CSI300_OUT", str(ROOT / "data" / "csi300_model_comparison.csv")))


def _summarize(preds_list: list[pd.DataFrame], label: str) -> dict:
    model_sh, em_sh, margins = [], [], []
    for p in preds_list:
        a = p["action"].to_numpy()
        m = p["market_ret"].to_numpy()
        ms = sharpe_ratio(a * m)
        es = sharpe_ratio(np.abs(a) * m)
        model_sh.append(ms)
        em_sh.append(es)
        margins.append(ms - es)
    return {
        "model": label,
        "sharpe_model_mean": round(float(np.mean(model_sh)), 4),
        "sharpe_model_std": round(float(np.std(model_sh)), 4),
        "sharpe_em_mean": round(float(np.mean(em_sh)), 4),
        "sharpe_em_std": round(float(np.std(em_sh)), 4),
        "margin_mean": round(float(np.mean(margins)), 4),
        "wins_vs_em": int(sum(np.array(model_sh) > np.array(em_sh))),
    }


def main() -> None:
    rows = []

    # ---- Model B: DDR (naive_new + vol-targeted) ----
    ddr_data = load_ddr_data(20)
    ddr_root = Path(os.environ.get(
        "CSI300_DDR_CKPT",
        str(ROOT / "src" / "models" / "ddr" / "checkpoints"),
    ))
    for run, label in [("naive_new", "B (DDR-naive)"), ("vt", "B-vt (DDR-voltarget)")]:
        preds = []
        for seed in SEEDS:
            cfg = replace(
                DDRConfig(),
                checkpoint_dir=ddr_root
                / run / f"s{seed}",
            )
            p = ddr_roll(cfg, checkpoint=cfg.checkpoint_dir / "ddr_best.pt", data=ddr_data)
            preds.append(p[p["valid"]].copy())
        rows.append(_summarize(preds, label))

    # ---- Model C: TACR ----
    tacr_cfg = TACRConfig.from_yaml()
    tacr_base = os.environ.get("CSI300_TACR_CKPT")
    if tacr_base:
        tacr_cfg = replace(tacr_cfg, checkpoint_dir=Path(tacr_base))
    tacr_data = load_tacr_data(tacr_cfg.u, exclude_policies=tacr_cfg.exclude_policies)
    tacr_splits = split_tacr_data(tacr_data)
    tacr_test_dates = tacr_splits["test"].dates
    tacr_preds = []
    tacr_abs_a = []
    for seed in SEEDS:
        cc = replace(tacr_cfg, seed=seed, checkpoint_dir=tacr_cfg.checkpoint_dir / f"s{seed}")
        p = tacr_roll(cc, checkpoint=cc.checkpoint_dir / "tacr_best.pt", data=tacr_data)
        pv = p[p["valid"]].copy()
        tacr_preds.append(pv)
        tacr_abs_a.append(np.abs(pv["action"].to_numpy()))
    rows.append(_summarize(tacr_preds, "C (TACR)"))

    # ---- Model D + D-minus-fuzzy ----
    d_cfg = DConfig.from_yaml()
    d_base = os.environ.get("CSI300_D_CKPT")
    if d_base:
        d_cfg = replace(d_cfg, checkpoint_dir=Path(d_base))
    for variant, label in [("D", "D (fuzzy+IQL)"), ("D-minus-fuzzy", "D-minus-fuzzy")]:
        preds = []
        for seed in SEEDS:
            p = roll_test_preds(d_cfg, variant, seed, "d_best.pt")
            preds.append(p[p["valid"]].copy())
        rows.append(_summarize(preds, label))

    # ---- C+ hybrid: TACR magnitude x linear sign (z-only and z+macro) ----
    # Reuse the sign models + actions from hybrid_sign_macro_cn by recomputing.
    from src.data.loaders import REPO_ROOT  # noqa: E402
    from src.data.macro_factors_cn import MACRO_FEATURES_CN, macro_features_cn, zscore_causal  # noqa: E402
    from src.models import SPLIT_TRAIN_END, SPLIT_VAL_END, SPLIT_TEST_END  # noqa: E402
    from sklearn.linear_model import LogisticRegression

    feats = pd.read_parquet(REPO_ROOT / "data" / "processed" / "features_regimes.parquet")
    csi_z = feats[["z_ret_1d", "z_ret_5d", "z_ret_20d", "z_realized_vol_20d",
                   "z_rsi_14", "z_macd_hist", "z_volume_zscore_20d", "z_bollinger_pos"]] \
        .reindex(tacr_data.dates).to_numpy(dtype="float64")
    macro_raw = macro_features_cn(feats)
    naive = tacr_data.dates.tz_localize(None)
    macro_z = np.column_stack([zscore_causal(macro_raw[c]) for c in MACRO_FEATURES_CN])
    macro_z = macro_z[np.searchsorted(macro_raw.index, naive)]
    Fb = np.concatenate([csi_z, macro_z], axis=1)

    dates = tacr_data.dates.tz_localize(None)
    tr = dates <= pd.Timestamp(SPLIT_TRAIN_END)
    va = (dates > pd.Timestamp(SPLIT_TRAIN_END)) & (dates <= pd.Timestamp(SPLIT_VAL_END))
    y = np.sign(tacr_data.market_returns.numpy()); y[y == 0] = 1
    okz = np.isfinite(csi_z).all(1)
    okm = np.isfinite(Fb).all(1)

    def _fit(Xtr, ytr, Xv, yv):
        best = (None, -1.0, None)
        for C in [0.01, 0.1, 1.0, 10.0]:
            m = LogisticRegression(penalty="l2", C=C, max_iter=2000).fit(Xtr, ytr)
            a = float((m.predict(Xv) == yv).mean())
            if a > best[1]:
                best = (m, a, C)
        return best[0]

    m_a = _fit(csi_z[tr & okz], y[tr & okz], csi_z[va & okz], y[va & okz])
    m_b = _fit(Fb[tr & okm], y[tr & okm], Fb[va & okm], y[va & okm])

    for name, Xm, model in [("C+ (hybrid z-only)", csi_z, m_a),
                            ("C+ (hybrid z+macro)", Fb, m_b)]:
        preds = []
        for seed in SEEDS:
            p = tacr_preds[SEEDS.index(seed)].copy()
            pos = tacr_data.dates.get_indexer(pd.DatetimeIndex(p["date"]))
            Xt = Xm[pos]
            good = np.isfinite(Xt).all(1)
            sig = model.predict(Xt[good])
            p = p[good].copy()
            p["action"] = sig * np.abs(p["action"].to_numpy())
            preds.append(p)
        rows.append(_summarize(preds, name))

    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT, index=False)
    print(tbl.round(4).to_string(index=False))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
