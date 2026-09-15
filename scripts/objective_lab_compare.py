"""Cross-objective representation comparison (idea 16).

Loads the four trained Objective Lab arms, extracts h_t from the SHARED
encoder on the full DDR grid, and runs the identical battery used for the
B/C/D models:

  linear probes (direction/magnitude/vol/regime/reconstruction)
  effective + spectral rank
  linear CKA across arms
  objective outcome (test Sharpe for the RL arms, test error for the
  supervised arms) — to tie representation content back to behavior.

Outputs under data/interpret/: objective_probe_table.csv,
objective_rank.csv, objective_cka.csv, objective_outcomes.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_representations import (
    PROBES, OUT_DIR, raw_window, reconstruction_rsq, run_probe,
)
from scripts.representation_rank import eff_rank
from src.interpret.targets import Z_COLUMNS, _naive, align, build_targets
from src.models.objective_lab.config import ARMS, ObjectiveLabConfig, tag_path
from src.models.objective_lab.train import extract_arm, load_arm


def objective_outcomes(cfg: ObjectiveLabConfig) -> pd.DataFrame:
    import torch
    from src.models.ddr.data import load_ddr_data, split_ddr_data
    from src.models.objective_lab.train import roll_sharpe

    data = load_ddr_data(cfg.window)
    test = split_ddr_data(data)["test"]
    mask = test.valid if test.valid is not None else torch.ones(len(test.windows), dtype=torch.bool)
    rows = []
    for arm in ARMS:
        ck = tag_path(cfg, arm) / "best.pt"
        if not ck.exists():
            continue
        agent = load_arm(arm, cfg, ck)
        with torch.no_grad():
            if arm == "A_predictive":
                pred = agent.head(agent.encoder.encode(test.windows[mask]))
                y = test.next_returns[mask]
                tmean = float(y.mean()); tstd = float(y.std()) + 1e-8
                yhat = pred.numpy() * tstd + tmean
                mse = float(np.mean((yhat - y.numpy()) ** 2))
                corr = float(np.corrcoef(yhat, y.numpy())[0, 1])
                rows.append({"arm": arm, "test_metric": -np.sqrt(mse),
                             "label": f"test RMSE {np.sqrt(mse) * 1e4:.0f} bps, corr {corr:+.3f}"})
            elif arm in ("B_dsr", "C_actorcritic"):
                sh = np.isfinite(roll_sharpe(agent, test, cfg.cost_bps)) or float("nan")
                sh = float(roll_sharpe(agent, test, cfg.cost_bps))
                rows.append({"arm": arm, "test_metric": sh, "label": f"test Sharpe {sh:.3f}"})
            else:
                pred = agent.head(agent.encoder.encode(test.windows[mask]))
                mse = float(np.mean((pred.numpy() - test.windows[mask][:, -1, :].numpy()) ** 2))
                rows.append({"arm": arm, "test_metric": -np.sqrt(mse),
                             "label": f"recon RMSE {np.sqrt(mse):.3f} (z-units)"})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = ObjectiveLabConfig()
    sub = build_targets()
    sub = sub[~_naive(sub.index).duplicated(keep="first")].copy()
    sub.index = _naive(sub.index)

    outs = objective_outcomes(cfg)
    outs.to_csv(OUT_DIR / "objective_outcomes.csv", index=False)
    print("\n=== objective outcomes (test split) ===")
    print(outs[["arm", "label"]].to_string(index=False))

    reps = []
    for arm in ARMS:
        ck = tag_path(cfg, arm) / "best.pt"
        if not ck.exists():
            print(f"[skip] {arm} not trained yet -> {ck}")
            continue
        rep = extract_arm(arm, cfg, ck)
        reps.append(rep)

    # ---- probe battery across arms ----
    rows = []
    for rep in reps:
        rsub, X = align(rep.dates, rep.H, sub)
        for probe_name, kind, col in PROBES:
            r = run_probe(X, rsub, probe_name, kind, col)
            if r:
                rows.append(dict(representation=rep.name, **r))
        rows.append(dict(representation=rep.name, probe="reconstruct", val=float("nan"),
                         test=reconstruction_rsq(X, rsub), baseline_test=0.0,
                         n_train=int((rsub["split"] == "train").sum()),
                         n_test=int((rsub["split"] == "test").sum())))
    pdf = pd.DataFrame(rows)
    pdf.to_csv(OUT_DIR / "objective_probe_results.csv", index=False)
    table = pdf.pivot_table(index="representation", columns="probe", values="test")
    table = table.reindex(columns=[p for p, _, _ in PROBES] + ["reconstruct"])
    table.to_csv(OUT_DIR / "objective_probe_table.csv")
    print("\n=== objective-lab linear probes (test) ===")
    print(table.round(3).to_string())

    # ---- rank ----
    rrows = []
    for rep in reps:
        rsub, X = align(rep.dates, rep.H, sub)
        m = {"representation": rep.name}
        m.update(eff_rank(X[(rsub["split"] == "test").to_numpy()]))
        rrows.append(m)
    rk = pd.DataFrame(rrows)
    rk.to_csv(OUT_DIR / "objective_rank.csv", index=False)
    print("\n=== effective / spectral rank (test) ===")
    print(rk[["representation", "effective_rank", "spectral_rank", "top1", "scale"]].round(3).to_string(index=False))

    # ---- CKA across arms ----
    from scripts.rep_sensitivity import linear_cka
    mats = {}
    for rep in reps:
        rsub, X = align(rep.dates, rep.H, sub)
        mats[rep.name] = X[(rsub["split"] == "test").to_numpy()]
    names = list(mats)
    links = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            nrow = min(len(mats[names[i]]), len(mats[names[j]]))
            links.append(dict(repr_a=names[i], repr_b=names[j],
                              cka=linear_cka(mats[names[i]][:nrow], mats[names[j]][:nrow])))
    cka = pd.DataFrame(links)
    cka.to_csv(OUT_DIR / "objective_cka.csv", index=False)
    print("\n=== linear CKA across objectives (test) ===")
    print(cka.round(4).to_string())
    print("\noutputs under", OUT_DIR)


if __name__ == "__main__":
    main()