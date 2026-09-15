"""Representation sensitivity + similarity (next_experiments.txt):
temporal receptive field, feature permutation-importance, per-feature
counterfactual responses, and linear CKA similarity between models.

All on the FROZEN test split (seed 20260814). For each perturbaation of a
model's causal window we measure:

  dh      = mean ||h_pert - h_base||_2      (how much the hidden state moves)
  dh_rel  = mean ||h_pert - h_base|| / sigma_h  (movement per h std)
  da      = mean |a_pert - a_base|          (action movement; a in [-1,1])
  da_sgn  = mean(a_pert - a_base)           (sign symmetry of the response)

Battery:
  temporal mask   zero the state slice lagged L days (L = 1..15)
  feature zero    zero one of the 8 canonical features across the window
  feature perm    shuffle one canonical feature across the window's TIME axis
  counterfactual  set last-day feature to its mean +1 or -1 (z-space)

Model-D's input is fuzzy 26-d (18 derived dims), so its feature battery is
restricted to temporal masks + a whole-state last-day counterfactual; the
canonical-feature tests apply to the 8-d DDR and TACR.

Outputs:
  sensitivity_scores.csv   per (model, perturbation, unit) mean responses
  cka_similarity.csv       linear CKA between all representation pairs
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
    extract_d, extract_ddr, extract_tacr, tacr_action_from_h,
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

S_SAMPLE = 300              # sampled test rows per model
LAGS = [1, 2, 3, 5, 10, 15]  # causal lags (days) to mask
CANONICAL = ["ret_1d", "ret_5d", "ret_20d", "rv20", "rsi", "macd",
             "volz", "boll"]


def test_rows(dates: pd.DatetimeIndex, sub: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Row positions (in the model's own sequence) that are TEST days,
    sub-sampled to <= n evenly. Returns (positions, naive dates);"""
    ad = np.array([d for d in _naive(dates).strftime("%Y-%m-%d")])
    test_set = set(_naive(sub.index[sub["split"] == "test"]).strftime("%Y-%m-%d"))
    rows = np.array([i for i, d in enumerate(ad) if d in test_set], dtype=int)
    if len(rows) > n:
        rows = rows[np.linspace(0, len(rows) - 1, n).astype(int)]
        rows = np.unique(rows)
    return rows, ad[rows]


def _stats(h_base: torch.Tensor, a_base: np.ndarray,
           h_pert: torch.Tensor, a_pert: np.ndarray) -> dict:
    d = (h_pert - h_base)
    dh = float(d.norm(dim=1).mean())
    sigma = float((h_base - h_base.mean(0)).norm(dim=1).mean())
    da = float(np.mean(np.abs(a_pert - a_base)))
    return {"dh": dh, "dh_rel": dh / sigma if sigma > 0 else float("nan"),
            "da": da, "da_sgn": float(np.mean(a_pert - a_base))}


def run_ddr(sub, cfg, ckpt) -> list:
    data = load_ddr_data(cfg.window_size)
    model = DDRPolicy(input_dim=data.windows.shape[2], hidden_size=cfg.hidden_size,
                      rnn_type=cfg.rnn_type)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    model.eval()
    W = data.windows.float()
    rows, _ = test_rows(data.dates, sub, S_SAMPLE)
    Ws = W[rows]
    with torch.no_grad():
        hb = model.rnn(Ws)[0][:, -1, :]
    ab = tacr_action_from_h(model, hb)
    out = []
    for lag in LAGS:
        Wp = Ws.clone(); Wp[:, Ws.shape[1] - 1 - lag, :] = 0.0
        with torch.no_grad():
            hp = model.rnn(Wp)[0][:, -1, :]
        m = _stats(hb, ab, hp, tacr_action_from_h(model, hp))
        out.append(dict(model="DDR", perturbation="mask_lag", unit=f"lag{lag}", **m))
    for j, f in enumerate(CANONICAL):
        Wp = Ws.clone(); Wp[:, :, j] = 0.0
        with torch.no_grad():
            hp = model.rnn(Wp)[0][:, -1, :]
        out.append(dict(model="DDR", perturbation="zero_feature", unit=f,
                        **_stats(hb, ab, hp, tacr_action_from_h(model, hp))))
        perm = torch.from_numpy(np.stack([np.random.RandomState(0).permutation(Ws.shape[1])
                                          for _ in range(len(Ws))]))
        Wp = Ws.clone(); Wp[:, :, j] = Wp[:, :, j].gather(1, perm).view(-1, Ws.shape[1])
        with torch.no_grad():
            hp = model.rnn(Wp)[0][:, -1, :]
        out.append(dict(model="DDR", perturbation="perm_feature", unit=f,
                        **_stats(hb, ab, hp, tacr_action_from_h(model, hp))))
        for sgn in (+1.0, -1.0):
            Wp = Ws.clone(); Wp[:, Ws.shape[1] - 1, j] = float(sgn)
            with torch.no_grad():
                hp = model.rnn(Wp)[0][:, -1, :]
            out.append(dict(model="DDR", perturbation=f"cf_last{sgn:+.0f}", unit=f,
                            **_stats(hb, ab, hp, tacr_action_from_h(model, hp))))
    return out


def run_tacr(sub, cfg, ckpt) -> list:
    data = load_tacr_data(cfg.u, exclude_policies=cfg.exclude_policies,
                          state_macro=cfg.state_macro)
    model, _, _ = load_checkpoint(ckpt, cfg)
    pred_actions = roll_actions(model, cfg, data, data.dates, cfg.rtg_target)
    ends = np.arange(len(data.dates))
    states_w = make_windows(data.states, ends, cfg.u)[0]
    actions_w = _make_windows_1d(pred_actions, ends, cfg.u)
    rtgs_w = torch.full_like(actions_w, cfg.rtg_target)
    ts_w = _make_windows_1d(data.timesteps[0].numpy(), ends, cfg.u, dtype=torch.long)

    def h_of(S: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return _tacr_state_tokens(model, S, actions_w[rows], rtgs_w[rows], ts_w[rows])[:, -1, :]

    rows, _ = test_rows(data.dates, sub, S_SAMPLE)
    Ws = states_w[rows].float()
    hb = h_of(Ws)
    out = []
    for lag in LAGS:
        Wp = Ws.clone(); Wp[:, Ws.shape[1] - 1 - lag, :] = 0.0
        out.append(dict(model="TACR", perturbation="mask_lag", unit=f"lag{lag}",
                        **_stats(hb, tacr_action_from_h(model, hb), h_of(Wp),
                                tacr_action_from_h(model, h_of(Wp)))))
    for j, f in enumerate(CANONICAL):
        Wp = Ws.clone(); Wp[:, :, j] = 0.0
        hpx = h_of(Wp)
        out.append(dict(model="TACR", perturbation="zero_feature", unit=f,
                        **_stats(hb, tacr_action_from_h(model, hb), hpx,
                                tacr_action_from_h(model, hpx))))
        perm = torch.from_numpy(np.stack([np.random.RandomState(0).permutation(Ws.shape[1])
                                          for _ in range(len(Ws))]))
        Wp = Ws.clone(); Wp[:, :, j] = Wp[:, :, j].gather(1, perm).view(-1, Ws.shape[1])
        hpx = h_of(Wp)
        out.append(dict(model="TACR", perturbation="perm_feature", unit=f,
                        **_stats(hb, tacr_action_from_h(model, hb), hpx,
                                tacr_action_from_h(model, hpx))))
        for sgn in (+1.0, -1.0):
            Wp = Ws.clone(); Wp[:, Ws.shape[1] - 1, j] = float(sgn)
            hpx = h_of(Wp)
            out.append(dict(model="TACR", perturbation=f"cf_last{sgn:+.0f}", unit=f,
                            **_stats(hb, tacr_action_from_h(model, hb), hpx,
                                    tacr_action_from_h(model, hpx))))
    return out


def run_d(sub, cfg, ckpt) -> list:
    data = load_d_data(cfg)
    agent = load_agent(ckpt, cfg)
    ends = np.arange(len(data.dates))
    windows, ts = make_windows(data.states_in, ends, cfg.u)
    rows, _ = test_rows(data.dates, sub, S_SAMPLE)
    Ws = windows[rows].float()
    Ts = ts[rows]
    with torch.no_grad():
        hb = agent.h_last(Ws, Ts)
        ab = agent.policy(hb).numpy()
    out = []
    for lag in LAGS:
        Wp = Ws.clone(); Wp[:, Ws.shape[1] - 1 - lag, :] = 0.0
        with torch.no_grad():
            hp = agent.h_last(Wp, Ts)
            out.append(dict(model="D", perturbation="mask_lag", unit=f"lag{lag}",
                            **_stats(hb, ab, hp, agent.policy(hp).numpy())))
    for sgn in (+1.0, -1.0):
        Wp = Ws.clone(); Wp[:, Ws.shape[1] - 1, :] = float(sgn)
        with torch.no_grad():
            hp = agent.h_last(Wp, Ts)
            out.append(dict(model="D", perturbation=f"cf_last{sgn:+.0f}", unit="state",
                            **_stats(hb, ab, hp, agent.policy(hp).numpy())))
    return out


# --------------------------------------------------------------------------
# CKA
# --------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0); Y = Y - Y.mean(0)
    Kx = X @ X.T; Ky = Y @ Y.T
    Kxc = Kx - Kx.mean(0, keepdims=True) - Kx.mean(1, keepdims=True) + Kx.mean()
    Kyc = Ky - Ky.mean(0, keepdims=True) - Ky.mean(1, keepdims=True) + Ky.mean()
    h1 = float(np.sqrt(np.sum(Kxc * Kxc)))
    h2 = float(np.sqrt(np.sum(Kyc * Kyc)))
    if h1 == 0 or h2 == 0:
        return float("nan")
    return float(np.sum(Kxc * Kyc) / (h1 * h2))


def run_cka(sub) -> pd.DataFrame:
    from src.interpret.targets import align
    ddr = extract_ddr(DDRConfig.from_yaml(), DDR_CKPT / f"s{SEED}" / "ddr_best.pt")
    tacr = extract_tacr(TACRConfig.from_yaml(), TACR_CKPT / f"s{SEED}" / "tacr_best.pt")
    d = extract_d(DConfig.from_yaml(), D_CKPT / f"s{SEED}" / "d_best.pt")
    mats = {}
    for rep in (ddr, tacr, d):
        rsub, X = align(rep.dates, rep.H, sub)
        mats[rep.name] = X[(rsub["split"] == "test").to_numpy()]
    names = list(mats)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            nrow = min(mats[names[i]].shape[0], mats[names[j]].shape[0])
            rows.append(dict(repr_a=names[i], repr_b=names[j],
                             cka=linear_cka(mats[names[i]][:nrow], mats[names[j]][:nrow])))
    return pd.DataFrame(rows)


def main() -> None:
    targets = build_targets()
    sub = targets[~_naive(targets.index).duplicated(keep="first")].copy()
    sub.index = _naive(sub.index)

    all_rows = []
    all_rows += run_ddr(sub, DDRConfig.from_yaml(), DDR_CKPT / f"s{SEED}" / "ddr_best.pt")
    print("DDR sensitivity done")
    all_rows += run_tacr(sub, TACRConfig.from_yaml(), TACR_CKPT / f"s{SEED}" / "tacr_best.pt")
    print("TACR sensitivity done")
    all_rows += run_d(sub, DConfig.from_yaml(), D_CKPT / f"s{SEED}" / "d_best.pt")
    print("D sensitivity done")

    scores = pd.DataFrame(all_rows)
    scores.to_csv(OUT_DIR / "sensitivity_scores.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== action-movement |da| by perturbation (means over sampled test rows) ===")
    pivot = scores.pivot_table(index=["model", "perturbation"], columns="unit",
                               values="da").loc[:, :]
    print(pivot.round(4).to_string())

    print("\n=== hidden-state movement dh_rel ===")
    pivot2 = scores.pivot_table(index=["model", "perturbation"], columns="unit",
                                values="dh_rel")
    print(pivot2.round(4).to_string())

    cka = run_cka(sub)
    cka.to_csv(OUT_DIR / "cka_similarity.csv", index=False)
    print("\n=== linear CKA (test split) ===")
    print(cka.round(4).to_string())
    print("\noutputs under", OUT_DIR)


if __name__ == "__main__":
    main()