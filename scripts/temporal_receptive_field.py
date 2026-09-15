"""Temporal receptive field + feature x time importance (next_experiments.txt):

Idea (lines 595-627): the causal context window defines a *temporal receptive
field* of the trading policy. Falsifiable claim: "TACR receives 20 days but
80% of its action variance comes from the last 3 days."

Part 1 - block-mask receptive field (lines 603-620):
  mask the state rows in contiguous blocks (e.g. days -1..-5, -6..-10,
  -11..-20) and measure how much the decision moves:
      I_k = E[(a_t - a_t^mask(k))^2]
  Also report the mean |da| and hidden-state movement.

Part 2 - feature x time importance (lines 630-655):
  for each canonical feature column and each of the 20 lag rows, zero the
  single cell and record action + hidden movement -> a t-1..t-20 grid per
  feature (the reference's importance matrix).

The context scan (u = 10/20/40/60) is *not* reproduced because it needs
retraining; this file only perturbs the FROZEN models (seed 20260814),
same protocol as rep_sensitivity.py.

Outputs:
  temporal_receptive_field.csv    per (model, block) I_k + action/hidden movement
  temporal_feature_x_time.csv     per (model, feature, lag) single-cell masks
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

from src.interpret.extract import (
    D_CKPT, DDR_CKPT, SEED, TACR_CKPT, _make_windows_1d, _tacr_state_tokens,
    tacr_action_from_h,
)
from src.interpret.targets import _naive, build_targets
from src.models.d.config import DConfig
from src.models.d.data import load_d_data, make_windows
from src.models.d.train import load_agent
from src.models.ddr.config import DDRConfig
from src.models.ddr.data import load_ddr_data
from src.models.ddr.policy import DDRPolicy
from src.models.tacr.config import TACRConfig
from src.models.tacr.data import load_tacr_data
from src.models.tacr.eval import load_checkpoint, roll_actions

OUT_DIR = ROOT / "data" / "interpret"
OUT_DIR.mkdir(parents=True, exist_ok=True)

S_SAMPLE = 300
CANONICAL = ["ret_1d", "ret_5d", "ret_20d", "rv20", "rsi", "macd",
             "volz", "boll"]


def test_rows(dates: pd.DatetimeIndex, sub: pd.DataFrame,
              n: int = S_SAMPLE) -> np.ndarray:
    """Row positions (model sequence) that are TEST days, evenly sub-sampled."""
    ad = np.array([d for d in _naive(dates).strftime("%Y-%m-%d")])
    test_set = set(_naive(sub.index[sub["split"] == "test"]).strftime("%Y-%m-%d"))
    rows = np.array([i for i, d in enumerate(ad) if d in test_set], dtype=int)
    if len(rows) > n:
        rows = rows[np.linspace(0, len(rows) - 1, n).astype(int)]
        rows = np.unique(rows)
    return rows


def block_slices(W: int) -> list[tuple[str, np.ndarray]]:
    """Reference blocks: last 5 days / previous 5 / oldest 10 / whole window."""
    return [
        ("last5d", np.arange(W - 5, W)),
        ("prior5d", np.arange(W - 10, W - 5)),
        ("oldest10d", np.arange(0, W - 10)),
        ("all20d", np.arange(0, W)),
    ]


def stats(h_base: torch.Tensor, a_base: np.ndarray,
          h_pert: torch.Tensor, a_pert: np.ndarray) -> dict:
    d = h_pert - h_base
    dh = float(d.norm(dim=1).mean())
    sigma = float((h_base - h_base.mean(0)).norm(dim=1).mean())
    da = np.abs(a_pert - a_base)
    return {
        "ik": float(np.mean(da ** 2)),
        "da": float(np.mean(da)),
        "dh": dh,
        "dh_rel": dh / sigma if sigma > 0 else float("nan"),
    }


def run_ddr(sub, cfg, ckpt) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_ddr_data(cfg.window_size)
    model = DDRPolicy(input_dim=data.windows.shape[2], hidden_size=cfg.hidden_size,
                      rnn_type=cfg.rnn_type)
    model.load_state_dict(torch.load(ckpt, map_location="cpu",
                                     weights_only=False)["state_dict"])
    model.eval()
    W = data.windows.float()
    rows = test_rows(data.dates, sub)
    Ws = W[rows]
    with torch.no_grad():
        hb = model.rnn(Ws)[0][:, -1, :]
    ab = tacr_action_from_h(model, hb)
    ab = ab if isinstance(ab, np.ndarray) else np.asarray(ab)

    def h_of(X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.rnn(X)[0][:, -1, :]

    rows_b = []
    for name, sl in block_slices(Ws.shape[1]):
        Wp = Ws.clone(); Wp[:, sl, :] = 0.0
        hp = h_of(Wp)
        rows_b.append(dict(model="DDR", block=name,
                           **stats(hb, ab, hp, tacr_action_from_h(model, hp))))
    rows_c = []
    for j, f in enumerate(CANONICAL):
        for L in range(1, Ws.shape[1] + 1):
            Wp = Ws.clone(); Wp[:, Ws.shape[1] - L, j] = 0.0
            hp = h_of(Wp)
            s = stats(hb, ab, hp, tacr_action_from_h(model, hp))
            rows_c.append(dict(model="DDR", feature=f, lag=L, **s))
    return pd.DataFrame(rows_b), pd.DataFrame(rows_c)


def run_tacr(sub, cfg, ckpt) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_tacr_data(cfg.u, exclude_policies=cfg.exclude_policies,
                          state_macro=cfg.state_macro)
    model, _, _ = load_checkpoint(ckpt, cfg)
    pred_actions = roll_actions(model, cfg, data, data.dates, cfg.rtg_target)
    ends = np.arange(len(data.dates))
    states_w = make_windows(data.states, ends, cfg.u)[0]
    actions_w = _make_windows_1d(pred_actions, ends, cfg.u)
    rtgs_w = torch.full_like(actions_w, cfg.rtg_target)
    ts_w = _make_windows_1d(data.timesteps[0].numpy(), ends, cfg.u,
                            dtype=torch.long)
    rows = test_rows(data.dates, sub)
    Ws = states_w[rows].float()

    def h_of(X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return _tacr_state_tokens(model, X, actions_w[rows], rtgs_w[rows],
                                      ts_w[rows])[:, -1, :]

    hb = h_of(Ws)
    ab = tacr_action_from_h(model, hb)
    ab = ab if isinstance(ab, np.ndarray) else np.asarray(ab)

    rows_b = []
    for name, sl in block_slices(Ws.shape[1]):
        Wp = Ws.clone(); Wp[:, sl, :] = 0.0
        hp = h_of(Wp)
        rows_b.append(dict(model="TACR", block=name,
                           **stats(hb, ab, hp, tacr_action_from_h(model, hp))))
    rows_c = []
    for j, f in enumerate(CANONICAL):
        for L in range(1, Ws.shape[1] + 1):
            Wp = Ws.clone(); Wp[:, Ws.shape[1] - L, j] = 0.0
            hp = h_of(Wp)
            s = stats(hb, ab, hp, tacr_action_from_h(model, hp))
            rows_c.append(dict(model="TACR", feature=f, lag=L, **s))
    return pd.DataFrame(rows_b), pd.DataFrame(rows_c)


def run_d(sub, cfg, ckpt) -> pd.DataFrame:
    data = load_d_data(cfg)
    agent = load_agent(ckpt, cfg)
    ends = np.arange(len(data.dates))
    windows, ts = make_windows(data.states_in, ends, cfg.u)
    rows = test_rows(data.dates, sub)
    Ws = windows[rows].float()
    Ts = ts[rows]
    with torch.no_grad():
        hb = agent.h_last(Ws, Ts)
        ab = agent.policy(hb).numpy()

    def h_of(X: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
        with torch.no_grad():
            h = agent.h_last(X, Ts)
            return h, agent.policy(h).numpy()

    rows_b = []
    for name, sl in block_slices(Ws.shape[1]):
        Wp = Ws.clone(); Wp[:, sl, :] = 0.0
        hp, ap = h_of(Wp)
        rows_b.append(dict(model="D", block=name, **stats(hb, ab, hp, ap)))
    return pd.DataFrame(rows_b)


def main() -> None:
    targets = build_targets()
    sub = targets[~_naive(targets.index).duplicated(keep="first")].copy()
    sub.index = _naive(sub.index)

    blocks, cells = [], []
    for tag, run in (("DDR", run_ddr), ("TACR", run_tacr)):
        b, c = run(sub, globals()[tag + "Config"].from_yaml(),
                   globals()[tag + "_CKPT"] / f"s{SEED}" / f"{tag.lower()}_best.pt")
        blocks.append(b); cells.append(c)
        print(tag, "temporal blocks + feature x time done")
    blocks.append(run_d(sub, DConfig.from_yaml(), D_CKPT / f"s{SEED}" / "d_best.pt"))
    print("D temporal blocks done")

    bdf = pd.concat(blocks, ignore_index=True)
    bdf.to_csv(OUT_DIR / "temporal_receptive_field.csv", index=False)
    cdf = pd.concat(cells, ignore_index=True)
    cdf.to_csv(OUT_DIR / "temporal_feature_x_time.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== block-mask receptive field ===")
    print(bdf.pivot_table(index="model", columns="block",
                          values="ik").round(4).to_string())
    print("\n=== action movement |da| by block ===")
    print(bdf.pivot_table(index="model", columns="block",
                          values="da").round(4).to_string())
    print("\n=== hidden-state movement dh_rel by block ===")
    print(bdf.pivot_table(index="model", columns="block",
                          values="dh_rel").round(4).to_string())

    for m in cdf["model"].unique():
        g = cdf[cdf["model"] == m][["feature", "lag", "da"]].copy()
        g["lag"] = g["lag"].astype(str)
        print(f"\n=== {m}: feature x time |da| (columns = lag days back from t) ===")
        print(g.pivot_table(index="feature", columns="lag",
                            values="da").round(4).to_string())
        band = g.copy()
        band["band"] = np.select(
            [band["lag"].astype(int) <= 5, band["lag"].astype(int) <= 10],
            ["t-1..5", "t-6..10"], default="t-11..20")
        agg = band.groupby(["feature", "band"])["da"].mean().unstack()
        print(agg.round(4).to_string())
    print("\ncontext scan (u=10/20/40/60) intentionally omitted "
          "(requires retraining, reference lines 595-601)")
    print("\noutputs under", OUT_DIR)


if __name__ == "__main__":
    main()