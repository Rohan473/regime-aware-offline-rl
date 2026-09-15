"""Representation Lab measurement (Stage 1 x Stage 2):

For every representation learner (raw window / auto / predictive /
contrastive) freeze the encoder and report:

  - the measure-before-trading battery: direction_1 / magnitude_1 / vol_5 /
    vol_20 / regime probes + reconstruction R^2 + effective (spectral) rank
  - a downstream policy check: same A2C head trained on the frozen
    representation -> test Sharpe, and a 'Supervised' cell = a logistic-fit
    direction policy -> test Sharpe.

That produces the representation x policy matrix central to the research
design: does representation learning help, and can RL compensate for a bad
representation?

Outputs under data/interpret/:
  rep_measure.csv      info battery per representation
  rep_matrix.csv       representation x {Supervised, RL-A2C} test Sharpe
  rep_cka.csv          linear CKA between learned representations
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_representations import (
    PROBES, OUT_DIR, reconstruction_rsq, run_probe,
)
from scripts.representation_rank import eff_rank
from scripts.rep_sensitivity import linear_cka
from src.interpret.targets import _naive, align, build_targets
from src.models.ddr.data import load_ddr_data
from src.models.rep_lab.config import OBJECTIVES, RepLabConfig, tag_path
from src.models.rep_lab.model import RawFeat, DownstreamAgent
from src.models.rep_lab.train import (
    BEST, downstream_sharpe, extract_rep, load_rep, raw_window_rep,
    supervised_direction_sharpe,
)

REPS = ["raw"] + list(OBJECTIVES)


def rep_H(name: str, cfg: RepLabConfig):
    if name == "raw":
        return raw_window_rep(cfg)
    return extract_rep(name, cfg, tag_path(cfg, name) / BEST)


def info_row(name: str, rep, sub) -> dict:
    rsub, X = align(rep.dates, rep.H, sub)
    row = {"representation": rep.name}
    for probe_name, kind, col in PROBES:
        if probe_name in ("direction_5", "direction_20"):
            continue
        r = run_probe(X, rsub, probe_name, kind, col)
        row[probe_name] = r["test"] if r else float("nan")
    row["reconstruct"] = reconstruction_rsq(X, rsub)
    row.update(eff_rank(X[(rsub["split"] == "test").to_numpy()]))
    return row, rsub, X


def supervised_sharpe_for(rsub, X, r_next) -> float:
    return supervised_direction_sharpe(X, rsub, r_next)


def downstream_test_sharpe(name: str, cfg: RepLabConfig) -> float:
    ck = tag_path(cfg, name, subdir="downstream") / BEST
    if not ck.exists():
        return float("nan")
    data = load_ddr_data(cfg.window)
    test = split_for_test(data)
    encoder = None
    if name != "raw":
        encoder = load_rep(name, cfg, tag_path(cfg, name) / BEST).encoder

    def feed(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return encoder.encode(x) if encoder is not None else x.reshape(x.shape[0], -1)

    agent = DownstreamAgent(cfg.hidden,
                            None if name != "raw" else RawFeat(cfg.window * cfg.in_dim, cfg.hidden))
    agent.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"])
    agent.eval()
    return downstream_sharpe(agent, feed, test, cfg)


def split_for_test(data):
    from src.models.ddr.data import split_ddr_data
    return split_ddr_data(data)["test"]


def main() -> None:
    cfg = RepLabConfig()
    sub = build_targets()
    sub = sub[~_naive(sub.index).duplicated(keep="first")].copy()
    sub.index = _naive(sub.index)

    info_rows, shapes, name_map = [], {}, {}
    for name in REPS:
        ck = tag_path(cfg, name) / BEST
        if name != "raw" and not ck.exists():
            print(f"[skip] representation {name} not trained yet -> {ck}")
            continue
        rep = rep_H(name, cfg)
        row, rsub, X = info_row(name, rep, sub)
        info_rows.append(row)
        shapes[name] = (rsub, X)
        name_map[name] = rep.name

    im = pd.DataFrame(info_rows)
    im.to_csv(OUT_DIR / "rep_measure.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== representation lab: measure before trading (test) ===")
    cols = ["representation", "direction_1", "magnitude_1", "vol_5", "vol_20",
            "regime", "reconstruct", "effective_rank", "spectral_rank"]
    print(im[cols].round(3).to_string(index=False))

    # ---- representation x policy matrix ----
    rows = []
    for name in REPS:
        if name not in shapes:
            continue
        rsub, X = shapes[name]
        r_next = rsub["fwd_ret_1"].to_numpy()
        sup = supervised_sharpe_for(rsub, X, r_next)
        rl = downstream_test_sharpe(name, cfg)
        rows.append({"representation": name_map[name], "Supervised": sup, "RL-A2C": rl})
    mx = pd.DataFrame(rows)
    mx.to_csv(OUT_DIR / "rep_matrix.csv", index=False)
    print("\n=== representation x policy test Sharpe ===")
    print(mx.round(3).to_string(index=False))

    # ---- CKA between learned representations (and raw) ----
    mats = {}
    for name in REPS:
        if name not in shapes:
            continue
        rsub, X = shapes[name]
        mats[name_map[name]] = X[(rsub["split"] == "test").to_numpy()]
    names = list(mats)
    links = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            nrow = min(len(mats[names[i]]), len(mats[names[j]]))
            links.append(dict(repr_a=names[i], repr_b=names[j],
                              cka=linear_cka(mats[names[i]][:nrow], mats[names[j]][:nrow])))
    ck = pd.DataFrame(links)
    ck.to_csv(OUT_DIR / "rep_cka.csv", index=False)
    print("\n=== linear CKA (test) ===")
    print(ck.round(4).to_string(index=False))
    print("\noutputs under", OUT_DIR)


if __name__ == "__main__":
    main()