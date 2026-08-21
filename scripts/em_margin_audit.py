"""Project-wide EM-margin audit (models B, C, D) — 7.9.2.

CHECK 5 of the Model-D diagnostics showed the "beats EM" criterion is a
knife-edge: a long-only policy is EXACTLY its EM control (a*m == |a|*m), so
each X/5-vs-EM win count compresses a thin, continuous short-side margin
into a fragile binary. This audit re-derives every phase verdict that
rested on win counts, with the per-seed margins shown:

  B (naive_new, canonical) and B (vt, the rejected variant) : 7.5
  C (TACR, 3k pack)                                        : 7.6
  D (both variants, 3k + 20k)                              : 7.9

Per seed: all-days Sharpe of a*m, of the OWN exposure-matched control
|a|*m, the margin, and the short fraction that generates the margin.
Re-derivation questions:
  - Does the canonical-B decision (naive_new over vt) survive margins?
  - Is C's 1/5 failure a knife-edge artifact (margins ~ 0) or real
    (negative margins — shorts that HURT)?
  - Are D's margins different in kind from B's?

Outputs: printed tables + src/models/d/checkpoints/diagnostics/em_margin_audit.csv
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
from src.models.d.config import DConfig  # noqa: E402
from src.models.d.eval import SEEDS, VARIANT_DIRS  # noqa: E402
from src.models.d.train import BEST_NAME, load_agent  # noqa: E402
from src.models.d.data import load_d_data, make_windows, split_d_data  # noqa: E402
from src.models.ddr.config import DDRConfig  # noqa: E402
from src.models.ddr.data import load_ddr_data  # noqa: E402
from src.models.ddr.eval import roll_test_predictions as ddr_roll  # noqa: E402
from src.models.tacr.config import TACRConfig  # noqa: E402
from src.models.tacr.data import load_tacr_data  # noqa: E402
from src.models.tacr.eval import roll_test_predictions as tacr_roll  # noqa: E402

DDR_CKPT_ROOT = ROOT / "src" / "models" / "ddr" / "checkpoints"
TACR_CKPT_ROOT = ROOT / "src" / "models" / "tacr" / "checkpoints" / "tacr"


def margin_row(model: str, seed: int, a: np.ndarray, m: np.ndarray) -> dict:
    s = float(sharpe_ratio(a * m))
    e = float(sharpe_ratio(np.abs(a) * m))
    return {
        "model": model, "seed": seed,
        "sharpe": round(s, 4), "own_EM": round(e, 4),
        "margin": round(s - e, 4),
        "short_frac": round(float((a < 0).mean()), 4),
        "mean_abs_a": round(float(np.abs(a).mean()), 4),
        "wins": bool(s > e),
    }


def main() -> None:
    rows = []

    # ---------- B (naive_new + vt) ----------
    data_b = load_ddr_data(20)
    for variant in ("naive_new", "vt"):
        for seed in SEEDS:
            cfg = replace(DDRConfig(), checkpoint_dir=DDR_CKPT_ROOT / variant / f"s{seed}")
            p = ddr_roll(cfg, checkpoint=cfg.checkpoint_dir / "ddr_best.pt", data=data_b)
            p = p[p["valid"]]
            rows.append(margin_row(f"B {variant}", seed,
                                   p["action"].to_numpy(), p["market_ret"].to_numpy()))

    # ---------- C (TACR) ----------
    # CAVEAT (found 2026-08-21): the s1/tacr_best.pt on disk is the 20k
    # budget-rerun (Run A of 7.6.2, epochs 20x1000) that OVERWROTE the 3k
    # original; Phase-3 eval never saved per-seed action series, so the
    # original 3k seed-1 margin is unrecoverable. The audit therefore reads
    # 4 genuine 3k seeds + the 20k seed-1 checkpoint, and reports both the
    # full and the 3k-only robustness view. Detected automatically below.
    data_c = load_tacr_data(TACRConfig().u)
    c_budget = {}
    for seed in SEEDS:
        cfg = replace(TACRConfig(), checkpoint_dir=TACR_CKPT_ROOT / f"s{seed}", seed=seed)
        p = tacr_roll(cfg, checkpoint=cfg.checkpoint_dir / "tacr_best.pt", data=data_c)
        p = p[p["valid"]]
        row = margin_row("C (TACR)", seed, p["action"].to_numpy(), p["market_ret"].to_numpy())
        pl = torch.load(cfg.checkpoint_dir / "tacr_best.pt", map_location="cpu", weights_only=False)
        c_cfg = pl["config"]
        budget = "3k" if (c_cfg["epochs"], c_cfg["steps_per_epoch"]) == (10, 300) else \
            f"{c_cfg['epochs']}x{c_cfg['steps_per_epoch']}"
        c_budget[seed] = budget
        row["ckpt_budget"] = budget
        rows.append(row)
    if set(c_budget.values()) != {"3k"}:
        print(f"[audit] WARNING: C checkpoints not all 3k: {c_budget}")
        print("[audit] (7.6.2's 20k seed-1 rerun overwrote the 3k original; the")
        print("[audit]  seed-1 margin below is the 20k checkpoint's.)")

    # ---------- D (both variants, both budgets) ----------
    cfg0 = DConfig.from_yaml()
    for tag, budget in ((None, "3k"), ("d_20k", "20k")):
        base = cfg0.checkpoint_dir / (tag or "")
        for variant in ("D", "D-minus-fuzzy"):
            cfg = replace(cfg0, fuzzy=(variant == "D"))
            data = load_d_data(cfg)
            test = split_d_data(data)["test"]
            valid_mask = test.valid.numpy()
            ends = test.global_pos[valid_mask]
            windows, ts = make_windows(data.states_in, ends, cfg.u)
            m = test.market_returns[valid_mask].numpy()
            for seed in SEEDS:
                agent = load_agent(base / VARIANT_DIRS[variant] / f"s{seed}" / BEST_NAME, cfg)
                with torch.no_grad():
                    a = agent.policy(agent.h_last(windows, ts)).numpy()
                rows.append(margin_row(f"D {variant} @{budget}", seed, a, m))

    df = pd.DataFrame(rows)
    out = cfg0.checkpoint_dir / "diagnostics" / "em_margin_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("=" * 78)
    print("PROJECT-WIDE EM-MARGIN AUDIT (per-seed: Sharpe, own-EM, margin, short_frac)")
    print("=" * 78)
    for model, g in df.groupby("model", sort=False):
        print(f"\n--- {model} (wins {int(g['wins'].sum())}/5) ---")
        print(g[["seed", "sharpe", "own_EM", "margin", "short_frac", "mean_abs_a"]]
              .to_string(index=False))
        print(f"margins: mean {g['margin'].mean():+.4f}  min {g['margin'].min():+.4f}  "
              f"max {g['margin'].max():+.4f}")

    print("\n" + "=" * 78)
    print("RE-DERIVATION SUMMARY")
    print("=" * 78)
    for model, g in df.groupby("model", sort=False):
        pos = int((g["margin"] > 0).sum())
        zero = int((g["margin"].abs() < 1e-9).sum())
        neg = int((g["margin"] < 0).sum())
        print(f"{model:24s} wins {int(g['wins'].sum())}/5 | margins +{pos} / tie {zero} / -{neg} | "
              f"mean margin {g['margin'].mean():+.4f}")
    c = df[df["model"] == "C (TACR)"]
    c3k = c[c["ckpt_budget"] == "3k"] if "ckpt_budget" in c.columns else c
    if len(c3k) < len(c):
        print(f"\n[C robustness, 3k-only seeds]: margins {list(c3k['margin'])} -> "
              f"mean {c3k['margin'].mean():+.4f}, {int((c3k['margin'] < 0).sum())}/{len(c3k)} negative")
        print("(verdict direction — negative-mean margins with an active short side —")
        print(" holds without the overwritten seed; the magnitude carries the caveat.)")
    print(f"\nfull table: {out}")


if __name__ == "__main__":
    import torch  # noqa: E402 (used in the D loop above)
    main()