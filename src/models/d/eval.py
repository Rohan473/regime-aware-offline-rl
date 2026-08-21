"""Model D evaluation — regime table, basin screening, and the pre-registered
screen / escalation / ablation verdicts (Phase-4 spec section 0).

Reuses ``src/eval/regime_eval.py`` (regime-conditional Sharpe PRIMARY,
blended secondary) exactly as B/C. Rolls are deterministic (tanh policy
mean) and BATCHED: at each decision date the encoder consumes the causal
window ending at that date — no autoregressive action feedback, so a roll is
one forward pass per split. There is no return-to-go channel, so no realized
future return can enter the context (the Model-C leakage class is
structurally absent here).

Outputs (under ``cfg.checkpoint_dir``, per variant subdir):
- regime_eval.csv          per-seed regime breakdown (n_days, sharpe, cum, dd,
                           exposure, turnover) for BEST and FINAL checkpoints
- vs_model_b.csv           B vs C vs D vs EM comparison table
- basin_screening.csv      per-seed best-val epoch + cross-seed test corr
- screen_summary.csv       the pre-registered 3k-screen / escalation / ablation
                           verdicts + final-epoch sanity check

The pre-registered rules (written in PROJECT_NOTES 7.8 / spec section 0)
are applied here WITHOUT post-hoc adjustment:
  screen bar      : beats EM on all-days Sharpe in >= 3/5 seeds
  escalation      : 3k -> 20k if EITHER variant clears the screen
  ablation verdict: |mean(D) - mean(D-minus-fuzzy)| > max(1 pooled SD,
                    naive_new seed spread, 0.24)
  basin screening : anomalously late best-val epoch OR cross-seed test
                    correlation < 0.7 (calibrated to D's OWN pack; the
                    B-calibrated 0.7 threshold did not transfer to C).
  final-epoch check: final checkpoint rolls evaluated alongside best-val
                    (standing protocol from PROJECT_NOTES 7.6).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import regime_eval  # noqa: E402
from src.models.d.config import DConfig  # noqa: E402
from src.models.d.data import load_d_data, split_d_data  # noqa: E402
from src.models.d.train import BEST_NAME, FINAL_NAME, load_agent, roll_split  # noqa: E402
from src.models.tacr.eval import model_b_and_em  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
VARIANT_DIRS = {"D": "d", "D-minus-fuzzy": "d_minus_fuzzy"}
BASIN_EPOCH_FRAC = 0.35   # good-basin runs peak in the first ~third of the schedule
CORR_FLAG = 0.7           # cross-seed test-prediction corr below this => suspect
NAIVE_NEW_SPREAD = 0.24   # Model B (naive_new) all-days Sharpe seed spread 0.99 +- 0.24
CRISIS_N_DAYS = 15        # crisis test days — directionally suggestive only


def roll_test_preds(
    cfg: DConfig,
    variant: str,
    seed: int,
    checkpoint_name: str,
    data_full=None,
) -> pd.DataFrame:
    """Deterministic test roll -> DataFrame(date, action, market_ret, ret,
    regime, valid) matching B/C's eval format."""
    c = replace(cfg, seed=seed, checkpoint_dir=cfg.checkpoint_dir / VARIANT_DIRS[variant] / f"s{seed}")
    cp = c.checkpoint_dir / checkpoint_name
    if not cp.exists():
        raise FileNotFoundError(f"no checkpoint at {cp} — run `python -m src.models.d.train` first")
    agent = load_agent(cp, c)
    # data must be loaded with THIS variant's input width: the checkpoint's
    # state_in_dim governs the agent, and the same flag governs the loader.
    data_cfg = replace(cfg, fuzzy=(variant == "D"))
    if data_full is None:
        data_full = load_d_data(data_cfg)
    test = split_d_data(data_full)["test"]
    dates, a, m = roll_split(agent, c, data_full, test)
    preds = pd.DataFrame(
        {
            "date": dates,
            "action": a,
            "market_ret": m,
            "ret": a * m,
        }
    )
    preds["regime"] = test.regimes.to_numpy()
    preds["valid"] = np.ones(len(preds), dtype=bool)
    return preds


def exposure_and_turnover(a: np.ndarray) -> dict:
    return {
        "mean_abs_action": round(float(np.abs(a).mean()), 4),
        "short_frac": round(float((a < 0).mean()), 4),
        "mean_turnover": round(float(np.abs(np.diff(a)).mean()), 4),
    }


def regime_rows(preds: pd.DataFrame, seed: int, best_epoch: int) -> list[dict]:
    a = preds["action"].to_numpy()
    expo = exposure_and_turnover(a)
    rows = []
    for regime in ("bull", "bear", "crisis", "all"):
        mask = preds["regime"].to_numpy() == regime if regime != "all" else np.ones(len(preds), bool)
        r = preds["ret"].to_numpy()[mask]
        if len(r) < 2:
            continue
        rec = {
            "seed": seed,
            "regime": regime,
            "best_val_epoch": best_epoch,
            "n_days": int(len(r)),
            "sharpe": round(regime_eval.sharpe_ratio(r), 4),
            "cum": round(float(np.prod(1 + r) - 1), 5),
            "max_dd": round(regime_eval.max_drawdown(r), 5),
        }
        rec.update(expo)
        rows.append(rec)
    return rows


def variant_summary(cfg: DConfig, variant: str, checkpoint_name: str) -> dict:
    """Per-seed test rolls + regime rows + basin screening for one variant.

    Data is loaded with THIS variant's input width (fuzzy flag); the D and
    D-minus-fuzzy state matrices differ (26-d vs 8-d).
    """
    data_cfg = replace(cfg, fuzzy=(variant == "D"))
    data_full = load_d_data(data_cfg)
    preds_by_seed, rows, logs = {}, [], {}
    for seed in SEEDS:
        p = roll_test_preds(cfg, variant, seed, checkpoint_name, data_full=data_full)
        preds_by_seed[seed] = p
        log = pd.read_csv(cfg.checkpoint_dir / VARIANT_DIRS[variant] / f"s{seed}" / "training_log.csv")
        best_ep = int(log["val_sharpe"].idxmax()) + 1
        logs[seed] = (best_ep, len(log))
        rows.extend(regime_rows(p, seed, best_ep))

    A = np.vstack([preds_by_seed[s]["action"].to_numpy() for s in SEEDS])
    mean_corr = {
        s: float(np.nanmean([np.corrcoef(A[i], A[j])[0, 1] for j in range(5) if j != i]))
        for i, s in enumerate(SEEDS)
    }
    flags = {
        s: (logs[s][0] > round(BASIN_EPOCH_FRAC * logs[s][1])) or (mean_corr[s] < CORR_FLAG)
        for s in SEEDS
    }
    all_sharpe = {
        s: float(regime_eval.sharpe_ratio(preds_by_seed[s]["ret"].to_numpy())) for s in SEEDS
    }
    return {
        "preds": preds_by_seed,
        "rows": rows,
        "logs": logs,
        "mean_corr": mean_corr,
        "flags": flags,
        "all_sharpe": all_sharpe,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Model D (IQL, both variants).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None,
                        help="checkpoint subdir tag (escalation runs use e.g. d_20k)")
    args = parser.parse_args(argv)

    cfg = DConfig.from_yaml(args.config) if args.config else DConfig.from_yaml()
    if args.tag is not None:
        cfg.checkpoint_dir = cfg.checkpoint_dir / args.tag
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    all_rows = []
    for variant in VARIANT_DIRS:
        print(f"\n===== {variant} (best-val checkpoints) =====")
        s = variant_summary(cfg, variant, BEST_NAME)
        summary[variant] = s
        pd.DataFrame(s["rows"]).to_csv(
            cfg.checkpoint_dir / VARIANT_DIRS[variant] / "regime_eval.csv", index=False
        )
        pd.DataFrame(
            [{"seed": seed, "best_val_epoch": s["logs"][seed][0], "n_epochs": s["logs"][seed][1],
              "mean_corr_vs_pack": round(s["mean_corr"][seed], 4), "flagged": s["flags"][seed]}
             for seed in SEEDS]
        ).to_csv(cfg.checkpoint_dir / VARIANT_DIRS[variant] / "basin_screening.csv", index=False)
        all_rows.extend([{**r, "variant": variant} for r in s["rows"]])
        print(f"basin flags: { {s_: 'FLAG' if f else 'ok' for s_, f in s['flags'].items()} }")
        print(f"all-days Sharpe per seed: { {s_: round(v, 4) for s_, v in s['all_sharpe'].items()} }")

    print("\nCRISIS CAVEAT: 15 test days — directionally suggestive only, NOT a finding.")

    # --- vs Model B (naive_new), C (TACR), EM (same 5-seed protocol) ---
    b_rows = model_b_and_em()
    b = b_rows[b_rows["policy"] == "B (naive_new DDR)"].set_index("seed")
    em = b_rows[b_rows["policy"] == "EM (|a| * m)"].set_index("seed")
    b_sharpe = [float(regime_eval.sharpe_ratio(b.loc[s, "ret"])) for s in SEEDS]
    em_sharpe = [float(regime_eval.sharpe_ratio(em.loc[s, "ret"])) for s in SEEDS]
    c_csv = Path(__file__).resolve().parents[3] / "src" / "models" / "tacr" / "checkpoints" / "tacr" / "regime_eval.csv"
    if c_csv.exists():
        c_tab = pd.read_csv(c_csv)
        c_all = c_tab[c_tab["regime"] == "all"].set_index("seed")
        c_sharpe = [float(c_all.loc[s, "sharpe"]) for s in SEEDS if s in c_all.index]
    else:
        c_sharpe = None
        print("[eval] note: no C (TACR) regime_eval.csv found — C row omitted from comparison")

    vs = []
    for label, sharpe, expo in (
        ("B (naive_new DDR)", b_sharpe,
         [{"mean_abs_action": float(b.loc[s, "mean_abs_action"]),
           "short_frac": float(b.loc[s, "short_frac"]),
           "mean_turnover": float(b.loc[s, "mean_turnover"])} for s in SEEDS]),
        ("EM (|a| * m)", em_sharpe,
         [{"mean_abs_action": float(em.loc[s, "mean_abs_action"]),
           "short_frac": 0.0,
           "mean_turnover": float(em.loc[s, "mean_turnover"])} for s in SEEDS]),
    ):
        vs.append({"policy": label, "all_sharpe_mean": round(float(np.mean(sharpe)), 4),
                   "all_sharpe_std": round(float(np.std(sharpe)), 4), **expo[0]})
    if c_sharpe is not None:
        vs.append({"policy": "C (TACR)", "all_sharpe_mean": round(float(np.mean(c_sharpe)), 4),
                   "all_sharpe_std": round(float(np.std(c_sharpe)), 4)})
    for variant in VARIANT_DIRS:
        s = summary[variant]
        sh = list(s["all_sharpe"].values())
        expo = exposure_and_turnover(
            np.concatenate([s["preds"][seed]["action"].to_numpy() for seed in SEEDS])
        )
        vs.append({"policy": variant, "all_sharpe_mean": round(float(np.mean(sh)), 4),
                   "all_sharpe_std": round(float(np.std(sh)), 4), **expo})
    vs_df = pd.DataFrame(vs)
    vs_df.to_csv(cfg.checkpoint_dir / "vs_model_b.csv", index=False)
    print("\n=== vs B / C / EM / D (all-days Sharpe, 5 seeds) ===")
    print(vs_df.round(4).to_string(index=False))

    # --- pre-registered screen + escalation + ablation verdict ---
    # EM per seed = exposure-matched control using the VARIANT's OWN |a|
    # (same protocol as C's eval: sharpe(|a| * market_ret)).
    d_wins = {}
    for v, s in summary.items():
        wins = 0
        for seed in SEEDS:
            p = s["preds"][seed]
            own_em = float(regime_eval.sharpe_ratio(np.abs(p["action"].to_numpy())
                                                    * p["market_ret"].to_numpy()))
            wins += int(s["all_sharpe"][seed] > own_em)
        d_wins[v] = wins
    escalation = any(w >= 3 for w in d_wins.values())

    d_sharpe = list(summary["D"]["all_sharpe"].values())
    mf_sharpe = list(summary["D-minus-fuzzy"]["all_sharpe"].values())
    pooled_sd = float(np.sqrt((np.var(d_sharpe) + np.var(mf_sharpe)) / 2.0))
    delta = abs(float(np.mean(d_sharpe) - np.mean(mf_sharpe)))
    bar = max(pooled_sd, NAIVE_NEW_SPREAD)
    ablation_verdict = "MEANINGFUL" if delta > bar else "NULL (within noise floor)"

    screen = pd.DataFrame(
        [
            {"metric": "wins_vs_EM_3of5", "D": d_wins["D"], "D_minus_fuzzy": d_wins["D-minus-fuzzy"],
             "rule": ">= 3/5 to clear screen"},
            {"metric": "escalate_3k_to_20k", "D": int(d_wins["D"] >= 3),
             "D_minus_fuzzy": int(d_wins["D-minus-fuzzy"] >= 3),
             "rule": "escalate BOTH if EITHER clears (pre-registered)"},
            {"metric": "ablation_delta", "D": round(delta, 4),
             "D_minus_fuzzy": "", "rule": f"|delta| > max(pooled SD {pooled_sd:.3f}, "
                                          f"naive_new spread {NAIVE_NEW_SPREAD}) = {bar:.3f} -> {ablation_verdict}"},
        ]
    )
    print("\n=== pre-registered screen / escalation / ablation ===")
    print(screen.to_string(index=False))
    print(f"escalation triggered: {escalation} (if True, run `--tag d_20k` with 40x500 steps)")
    screen.to_csv(cfg.checkpoint_dir / "screen_summary.csv", index=False)

    # --- final-epoch sanity check (standing protocol) ---
    print("\n=== final-epoch sanity check (best-val vs final checkpoint) ===")
    rows = []
    for variant in VARIANT_DIRS:
        try:
            fs = variant_summary(cfg, variant, FINAL_NAME)
        except FileNotFoundError:
            print(f"[final-epoch] {variant}: no final checkpoint yet — skipped")
            continue
        f_sharpe = list(fs["all_sharpe"].values())
        b_sharpe_v = list(summary[variant]["all_sharpe"].values())
        rows.append({"variant": variant, "best_val_mean": round(float(np.mean(b_sharpe_v)), 4),
                     "final_mean": round(float(np.mean(f_sharpe)), 4),
                     "diff": round(float(np.mean(f_sharpe) - np.mean(b_sharpe_v)), 4)})
        print(f"{variant}: best-val mean {np.mean(b_sharpe_v):+.4f} vs final mean "
              f"{np.mean(f_sharpe):+.4f} (diff {np.mean(f_sharpe) - np.mean(b_sharpe_v):+.4f})")
    if rows:
        pd.DataFrame(rows).to_csv(cfg.checkpoint_dir / "final_epoch_check.csv", index=False)
    print(f"\nall outputs under {cfg.checkpoint_dir}")


if __name__ == "__main__":
    main()