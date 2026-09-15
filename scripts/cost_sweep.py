"""Net-of-transaction-cost margin sweep on the existing CSI300 checkpoints.

Pricing the verdict. All exposure-matched margins and the model-selection
ranking (data/csi300_model_comparison.csv) were computed at ZERO cost. This
script re-prices every model's test series (no retraining) with turnover-based
cost applied to BOTH the model and its own EM control, over a grid of cost
levels and a crisis-conditional multiplier, and reports whether the RANKING
flips.

Cost model (turnover, matching src/data/behavior_policies.py):
    model_t = a_t * m_t  - (bps/1e4) * |a_t - a_{t-1}|         a_{-1} = 0
    em_t    = |a_t| * m_t - (bps/1e4) * ||a_t| - |a_{t-1}||
Both accounts start flat (entry charged once). Buy-and-hold pays its entry
and nothing after; short-window rebalancers pay every flip; the EM long-only
sizing control rebalances only when its ABSOLUTE exposure changes.

Grid: flat bps in {0.0, 0.5, 1.0, 2.0, 5.0, 10.0}; crisis-boost sets (base
{1, 5} bps x5 on crisis days); and a CONTINUOUS vol-scaled robustness band:

    cost_bps_t = base_bps * (1 + k * vol_z_t)        clamped to >= 0

where vol_z_t is the shared z_realized_vol_20d feature (causal, the same
normalization the models saw as state input), for base in {0.5, 1, 2, 5} bps
x k in {0.5, 1.0, 2.0} (k=0 collapses to the flat grid). The vol-scaled rows
are NOT a cost model — they are a SENSITIVITY ASSUMPTION: they answer "does
the ranking survive under a plausible range of cost-widening-with-vol
behavior?", not "what cost actually was". At vol_z=+2 (high vol) the
multiplier is 2x-5x depending on k, the stress-widening band cited in the
flat-base calibration.

Output: data/cost_sweep.csv — one row per (model, bps, crisis_mult, vol_k).

Checkpoint roots are overridable via env vars (see compile_csi300_comparison):
  CSI300_DDR_CKPT  CSI300_TACR_CKPT  CSI300_D_CKPT  CSI300_OUT

Read-only: loads checkpoints, never trains, never writes checkpoints.
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.regime_eval import sharpe_ratio  # noqa: E402
from src.models.d.config import DConfig  # noqa: E402
from src.models.d.eval import roll_test_preds  # noqa: E402
from src.models.ddr.config import DDRConfig  # noqa: E402
from src.models.ddr.data import load_ddr_data  # noqa: E402
from src.models.ddr.eval import roll_test_predictions as ddr_roll  # noqa: E402
from src.models.tacr.config import TACRConfig  # noqa: E402
from src.models.tacr.data import load_tacr_data  # noqa: E402
from src.models.tacr.eval import roll_test_predictions as tacr_roll  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
OUT = Path(os.environ.get("CSI300_OUT", str(ROOT / "data" / "cost_sweep.csv")))

# Grid cells (bps, crisis_multiplier, vol_k). vol_k=0 = flat cost (no vol
# scaling): bps=0 reproduces the zero-cost baseline.
GRID: list[dict] = []
for bps in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0):
    GRID.append(dict(bps=bps, crisis_mult=1.0, vol_k=0.0))
GRID.append(dict(bps=1.0, crisis_mult=5.0, vol_k=0.0))
GRID.append(dict(bps=5.0, crisis_mult=5.0, vol_k=0.0))
for bps in (0.5, 1.0, 2.0, 5.0):
    for k in (0.5, 1.0, 2.0):
        GRID.append(dict(bps=bps, crisis_mult=1.0, vol_k=k))


def _naive_index(s: pd.Series) -> pd.Series:
    """Datetime series as naive (tz-free) index for joins."""
    d = pd.to_datetime(s)
    if getattr(getattr(d, "dt", None), "tz", None) is not None:
        d = d.dt.tz_localize(None)
    return d


def _vol_z_series() -> pd.Series | None:
    """Causal realized-vol z-score feature, index = naive daily dates."""
    p = ROOT / "data" / "processed" / "features_regimes.parquet"
    if not p.exists():
        return None
    feats = pd.read_parquet(p)
    if "z_realized_vol_20d" not in feats.columns:
        return None
    v = feats["z_realized_vol_20d"].astype("float64")
    v.index = _naive_index(pd.Series(feats.index)).to_numpy()
    return v


def net_returns(
    a: np.ndarray,
    m: np.ndarray,
    regime: np.ndarray,
    bps: float,
    crisis_mult: float,
    vol_k: float = 0.0,
    vol_z: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Net-of-cost model + EM return series (both start flat, a_{-1} = 0).

    Day-t cost multiplier = crisis mult * (1 + vol_k*vol_z), clamped >= 0.
    vol_k=0 (or missing vol_z) reduces to the flat/crisis grid.
    """
    mult = np.ones(len(a))
    if vol_k > 0 and vol_z is not None:
        mult = np.maximum(1.0 + vol_k * vol_z, 0.0)
    if crisis_mult != 1.0:
        mult = mult * np.where(regime == "crisis", crisis_mult, 1.0)
    cost_fn = (bps / 1e4) * mult
    da = np.abs(np.diff(a, prepend=0.0))
    model_ret = a * m - cost_fn * da
    absa = np.abs(a)
    em_ret = absa * m - cost_fn * np.abs(np.diff(absa, prepend=0.0))
    return model_ret, em_ret


def _summarize(
    preds_list: list[pd.DataFrame],
    label: str,
    bps: float,
    crisis_mult: float,
    vol_k: float,
    vol_z_src: pd.Series | None,
) -> dict:
    model_sh, em_sh, margins, wins = [], [], [], 0
    for p in preds_list:
        a = p["action"].to_numpy()
        m = p["market_ret"].to_numpy()
        regime = p["regime"].to_numpy()
        vol_z = None
        if vol_z_src is not None:
            idx = _naive_index(p["date"])
            vol_z = np.nan_to_num(
                vol_z_src.reindex(idx).to_numpy(dtype="float64"), nan=0.0
            )
        mr, er = net_returns(a, m, regime, bps, crisis_mult, vol_k, vol_z)
        ms = sharpe_ratio(mr)
        es = sharpe_ratio(er)
        model_sh.append(ms)
        em_sh.append(es)
        margins.append(ms - es)
        wins += int(ms > es)
    return {
        "model": label,
        "bps": bps,
        "crisis_mult": crisis_mult,
        "vol_k": vol_k,
        "sharpe_model_mean": round(float(np.mean(model_sh)), 4),
        "sharpe_model_std": round(float(np.std(model_sh)), 4),
        "sharpe_em_mean": round(float(np.mean(em_sh)), 4),
        "sharpe_em_std": round(float(np.std(em_sh)), 4),
        "margin_mean": round(float(np.mean(margins)), 4),
        "wins_vs_em": wins,
    }


def load_all_preds() -> OrderedDict[str, list[pd.DataFrame]]:
    """Roll every model's test predictions once; sweep costs on the fly."""
    preds_map: OrderedDict[str, list[pd.DataFrame]] = OrderedDict()

    # ---- Model B: DDR (naive_new + vol-targeted), zero cost at roll time ----
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
        preds_map[label] = preds

    # ---- Model C: TACR ----
    tacr_cfg = TACRConfig.from_yaml()
    tacr_base = os.environ.get("CSI300_TACR_CKPT")
    if tacr_base:
        tacr_cfg = replace(tacr_cfg, checkpoint_dir=Path(tacr_base))
    tacr_data = load_tacr_data(tacr_cfg.u, exclude_policies=tacr_cfg.exclude_policies)
    tacr_preds = []
    for seed in SEEDS:
        cc = replace(tacr_cfg, seed=seed, checkpoint_dir=tacr_cfg.checkpoint_dir / f"s{seed}")
        p = tacr_roll(cc, checkpoint=cc.checkpoint_dir / "tacr_best.pt", data=tacr_data)
        tacr_preds.append(p[p["valid"]].copy())
    preds_map["C (TACR)"] = tacr_preds

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
        preds_map[label] = preds

    # ---- C+ hybrid: TACR magnitude x linear sign (z-only and z+macro) ----
    from src.data.macro_factors_cn import (  # noqa: E402
        MACRO_FEATURES_CN,
        macro_features_cn,
        zscore_causal,
    )
    from src.data.loaders import REPO_ROOT  # noqa: E402
    from src.models import SPLIT_TRAIN_END, SPLIT_VAL_END  # noqa: E402
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
    y = np.sign(tacr_data.market_returns.numpy())
    y[y == 0] = 1
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
        preds_map[name] = preds

    return preds_map


def main() -> None:
    preds_map = load_all_preds()
    vol_z = _vol_z_series()
    if vol_z is None:
        print("[cost_sweep] WARNING: z_realized_vol_20d not found — "
              "vol-scaled rows collapse to flat; only flat/crisis grid reported.")

    rows = []
    for label, preds in preds_map.items():
        for cell in GRID:
            rows.append(_summarize(preds, label, **cell, vol_z_src=vol_z))

    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT, index=False)

    print("=== net-of-cost margins (model vs its OWN EM), CSI300 test ===")
    print(tbl.round(4).to_string(index=False))

    flat = tbl[tbl["vol_k"] == 0]
    print("\n=== RANKING (flat grid): highest-margin model at each cost level ===")
    for (bps, cm), g in flat.groupby(["bps", "crisis_mult"]):
        g = g[g["margin_mean"] > 0].sort_values("margin_mean", ascending=False)
        tag = "crisis-x5" if cm > 1 else ""
        if len(g):
            print(f"  {bps:>5.1f}bps {tag:<9} -> {g.iloc[0]['model']} "
                  f"(margin {g.iloc[0]['margin_mean']:+.3f}, wins {g.iloc[0]['wins_vs_em']}/5)")
        else:
            print(f"  {bps:>5.1f}bps {tag:<9} -> NO model keeps a positive margin")

    vol_rows = tbl[(tbl["vol_k"] > 0) & (tbl["crisis_mult"] == 1)]
    print("\n=== RANKING (vol-scaled band, base*(1+k*vol_z)): "
          "does the flat-grid winner survive? ===")
    for (bps, k), g in vol_rows.groupby(["bps", "vol_k"]):
        g = g[g["margin_mean"] > 0].sort_values("margin_mean", ascending=False)
        if len(g):
            print(f"  base={bps:>4.1f}bps k={k:.1f} -> {g.iloc[0]['model']} "
                  f"(margin {g.iloc[0]['margin_mean']:+.3f}, wins {g.iloc[0]['wins_vs_em']}/5)")
        else:
            print(f"  base={bps:>4.1f}bps k={k:.1f} -> NO model keeps a positive margin")

    zoom = tbl[tbl["model"].isin(["B-vt (DDR-voltarget)", "C+ (hybrid z-only)"])]
    print("\n=== B-vt vs C+ z-only margin gap across the grid ===")
    pivot = zoom.pivot_table(
        index=["bps", "crisis_mult"], columns=["vol_k", "model"],
        values="margin_mean",
    )
    print(pivot.round(3).to_string())

    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()