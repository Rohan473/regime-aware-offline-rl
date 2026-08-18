"""Model C (TACR) evaluation — shared with Model B's protocol.

Reuses ``src/eval/regime_eval.py`` (regime-conditional Sharpe as PRIMARY,
blended secondary — same axis as Model B). The roll is autoregressive:
at each decision date the actor conditions on the last u days of
(rtg=constant target, state, own predicted actions) and outputs one
action, which feeds the next context — exactly the paper's eval protocol
(evaluate_episodes feeds a zero return-to-go and rolls the model on its
own actions).

Fairness vs Model B: the constant rtg_target (default 0.0) is the ONLY
return channel at roll time — no realized future returns enter the
context, so TACR and DDR are compared on the same information set.

Outputs (checkpoints/tacr/):
- regime_eval.csv  — per-seed regime breakdown (n_days, sharpe, cum, dd, exposure, turnover)
- vs_model_b.csv   — the B-vs-C-vs-EM comparison table
- basin_screening.csv — per-seed best-val epoch + cross-seed test corr
"""

from __future__ import annotations

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
from src.models.ddr.config import DDRConfig  # noqa: E402
from src.models.ddr.data import load_ddr_data, split_ddr_data  # noqa: E402
from src.models.ddr.eval import roll_test_predictions as ddr_roll  # noqa: E402
from src.models.tacr.config import TACRConfig  # noqa: E402
from src.models.tacr.data import TACRData, load_tacr_data, split_tacr_data  # noqa: E402
from src.models.tacr.policy import TACRPolicy  # noqa: E402

SEEDS = [20260814, 1, 2, 3, 4]
BASIN_EPOCH_FRAC = 0.35  # good-basin runs peak in the first ~third of the schedule;
# anomalously late best-val epoch (Model B: 29/30 vs pack 3-5/30) => suspect basin
CORR_FLAG = 0.7        # cross-seed test-prediction correlation below this => suspect


def load_checkpoint(path: Path, cfg: TACRConfig) -> tuple[TACRPolicy, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = TACRPolicy(
        state_dim=payload["state_dim"],
        act_dim=payload["act_dim"],
        u=cfg.u,
        embed_dim=cfg.embed_dim,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_inner=cfg.n_inner,
        dropout=cfg.dropout,
        max_ep_len=cfg.max_ep_len,
        action_head=cfg.action_head,
    )
    model.load_state_dict(payload["actor"])
    model.eval()
    return model, payload


def roll_actions(
    model: TACRPolicy,
    cfg: TACRConfig,
    data: TACRData,
    split_dates: pd.DatetimeIndex,
    rtg_target: float,
) -> np.ndarray:
    """Autoregressive action roll over decision dates (paper's eval).

    At each date t the context is the last u triples ending at t:
    states = real day states (normalized), actions = the model's OWN past
    predictions, rtg = constant target, timesteps = absolute positions.
    Missing history before the series start is zero-padded (paper pads the
    same way in get_action).
    """
    u = cfg.u
    pos = {d: i for i, d in enumerate(data.dates)}
    T = len(data.dates)
    pred_actions = np.zeros(T)
    states = data.states.numpy()  # (T, F) shared across policies
    timesteps = data.timesteps[0].numpy()
    with torch.no_grad():
        for d in split_dates:
            p = pos[d]
            lo = max(0, p - u + 1)
            n = p - lo + 1
            pad = u - n
            st = torch.tensor(states[lo : p + 1], dtype=torch.float32).unsqueeze(0)
            ac = torch.tensor(pred_actions[lo : p + 1], dtype=torch.float32).unsqueeze(0)
            rt = torch.full((1, n), rtg_target, dtype=torch.float32)
            ts = torch.tensor(timesteps[lo : p + 1], dtype=torch.long).unsqueeze(0)
            if pad > 0:
                st = torch.cat([torch.zeros(1, pad, st.shape[-1]), st], dim=1)
                ac = torch.cat([torch.zeros(1, pad), ac], dim=1)
                rt = torch.cat([torch.zeros(1, pad), rt], dim=1)
                ts = torch.cat([torch.zeros(1, pad, dtype=torch.long), ts], dim=1)
            out = model(st, ac, rt, ts)
            pred_actions[p] = float(out[0, -1])
    return pred_actions


def roll_test_predictions(
    cfg: TACRConfig,
    checkpoint: Path | None = None,
    data: TACRData | None = None,
) -> pd.DataFrame:
    """Test-set roll -> DataFrame(date, action, market_ret, ret, regime, valid)
    matching Model B's eval format so regime_eval sees identical inputs."""
    if checkpoint is None:
        checkpoint = cfg.checkpoint_dir / "tacr_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint} — run `python -m src.models.tacr.train` first")
    model, _ = load_checkpoint(checkpoint, cfg)
    if data is None:
        data = load_tacr_data(cfg.u)
    splits = split_tacr_data(data)
    test = splits["test"]
    dates = test.dates

    actions = roll_actions(model, cfg, data, dates, cfg.rtg_target)
    # roll_actions returns the full-length array; extract the test dates
    actions = actions[data.dates.get_indexer(dates)]
    m = test.market_returns.numpy()
    preds = pd.DataFrame(
        {
            "date": dates,
            "action": actions,
            "market_ret": m,
            "ret": actions * m,
            "regime": test.regimes.to_numpy(),
            "valid": test.valid.numpy(),
        }
    )
    return preds


def exposure_and_turnover(preds: pd.DataFrame) -> dict:
    a = preds["action"].to_numpy()
    return {
        "mean_abs_action": round(float(np.abs(a).mean()), 4),
        "short_frac": round(float((a < 0).mean()), 4),
        "mean_turnover": round(float(np.abs(np.diff(a)).mean()), 4),
    }


def seed_roll(cfg: TACRConfig, seed: int, data: TACRData) -> pd.DataFrame:
    # cfg is already seed-scoped (checkpoint_dir/.../s{seed}) by the caller
    preds = roll_test_predictions(cfg, checkpoint=cfg.checkpoint_dir / "tacr_best.pt", data=data)
    return preds[preds["valid"]].copy()


def model_b_and_em() -> pd.DataFrame:
    """Model B (canonical naive_new) + its per-seed EM control, same protocol."""
    data = load_ddr_data(20)
    rows = []
    for seed in SEEDS:
        cfg = replace(
            DDRConfig(),
            checkpoint_dir=Path(__file__).parent.parent / "ddr" / "checkpoints" / "naive_new" / f"s{seed}",
        )
        preds = ddr_roll(cfg, checkpoint=cfg.checkpoint_dir / "ddr_best.pt", data=data)
        p = preds[preds["valid"]].copy()
        a = p["action"].to_numpy()
        m = p["market_ret"].to_numpy()
        rec = {
            "seed": seed,
            "policy": "B (naive_new DDR)",
            "ret": p["ret"].to_numpy(),
            "mean_abs_action": float(np.abs(a).mean()),
            "short_frac": float((a < 0).mean()),
            "mean_turnover": float(np.abs(np.diff(a)).mean()),
            "regime": p["regime"].to_numpy(),
        }
        rows.append(rec)
        rows.append(
            {
                "seed": seed,
                "policy": "EM (|a| * m)",
                "ret": np.abs(a) * m,
                "mean_abs_action": float(np.abs(a).mean()),
                "short_frac": 0.0,
                "mean_turnover": float(np.abs(np.diff(np.abs(a))).mean()),
                "regime": p["regime"].to_numpy(),
            }
        )
    return pd.DataFrame(rows)


def regime_row(ret: np.ndarray, regime: np.ndarray, exposure: dict) -> dict:
    row = {
        "sharpe": round(regime_eval.sharpe_ratio(ret), 4),
        "cum": round(float(np.prod(1 + ret) - 1), 5),
        "max_dd": round(regime_eval.max_drawdown(ret), 5),
        "n_days": int(len(ret)),
    }
    row.update(exposure)
    return row


def main() -> None:
    data = load_tacr_data(TACRConfig().u)
    splits = split_tacr_data(data)
    cfg = TACRConfig.from_yaml()
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=== per-seed test rolls + basin screening ===")
    preds_by_seed, rows, logs = {}, [], {}
    for seed in SEEDS:
        c = replace(cfg, seed=seed, checkpoint_dir=cfg.checkpoint_dir / f"s{seed}")
        p = seed_roll(c, seed, data)
        preds_by_seed[seed] = p
        log = pd.read_csv(c.checkpoint_dir / "training_log.csv")
        best_ep = int(log["val_sharpe"].idxmax()) + 1
        logs[seed] = (best_ep, len(log))
        expo = exposure_and_turnover(p)
        for regime in ("bull", "bear", "crisis", "all"):
            mask = p["regime"].to_numpy() == regime if regime != "all" else np.ones(len(p), bool)
            r = p["ret"].to_numpy()[mask]
            if len(r) < 2:
                continue
            rec = {"seed": seed, "regime": regime, "best_val_epoch": best_ep}
            rec.update(regime_row(r, p["regime"].to_numpy()[mask], expo))
            rows.append(rec)
        print(f"seed {seed}: best-val epoch {best_ep} (flag > {round(BASIN_EPOCH_FRAC * len(log))}), "
              f"all Sharpe {regime_eval.sharpe_ratio(p['ret'].to_numpy()):.4f}")

    # cross-seed correlation (test-day actions)
    A = np.vstack([preds_by_seed[s]["action"].to_numpy() for s in SEEDS])
    corr = pd.DataFrame(np.corrcoef(A), index=SEEDS, columns=SEEDS).round(4)
    print("cross-seed test-prediction correlation:")
    print(corr.to_string())
    mean_corr = {
        s: float(np.nanmean([np.corrcoef(A[i], A[j])[0, 1] for j in range(5) if j != i]))
        for i, s in enumerate(SEEDS)
    }
    flags = {
        s: (logs[s][0] > round(BASIN_EPOCH_FRAC * logs[s][1])) or (mean_corr[s] < CORR_FLAG)
        for s in SEEDS
    }
    print("basin flags:", {s: "FLAG" if f else "ok" for s, f in flags.items()})
    pd.DataFrame(
        [
            {"seed": s, "best_val_epoch": logs[s][0], "n_epochs": logs[s][1],
             "mean_corr_vs_pack": round(mean_corr[s], 4), "flagged": flags[s]}
            for s in SEEDS
        ]
    ).to_csv(cfg.checkpoint_dir / "basin_screening.csv", index=False)

    table = pd.DataFrame(rows)
    table.to_csv(cfg.checkpoint_dir / "regime_eval.csv", index=False)
    print("\n=== regime breakdown (TACR, 5 seeds) ===")
    print(
        table.pivot_table(index="regime", columns="seed", values="sharpe").round(4).to_string()
    )
    print("CRISIS CAVEAT: 15 test days — directionally suggestive only, NOT a finding.")

    print("\n=== vs Model B and EM (same 5-seed protocol) ===")
    b_rows = model_b_and_em()
    summary = []
    for policy, group in b_rows.groupby("policy"):
        sharpe_all = group["ret"].apply(regime_eval.sharpe_ratio).to_numpy()
        summary.append(
            {
                "policy": policy,
                "all_sharpe_mean": round(float(sharpe_all.mean()), 4),
                "all_sharpe_std": round(float(sharpe_all.std()), 4),
                "mean_abs_action": round(float(group["mean_abs_action"].mean()), 4),
                "short_frac": round(float(group["short_frac"].mean()), 4),
                "turnover": round(float(group["mean_turnover"].mean()), 4),
            }
        )
    tacr_sharpe = [regime_eval.sharpe_ratio(preds_by_seed[s]["ret"].to_numpy()) for s in SEEDS]
    tacr_expo = [exposure_and_turnover(preds_by_seed[s]) for s in SEEDS]
    summary.append(
        {
            "policy": "C (TACR)",
            "all_sharpe_mean": round(float(np.mean(tacr_sharpe)), 4),
            "all_sharpe_std": round(float(np.std(tacr_sharpe)), 4),
            "mean_abs_action": round(float(np.mean([e["mean_abs_action"] for e in tacr_expo])), 4),
            "short_frac": round(float(np.mean([e["short_frac"] for e in tacr_expo])), 4),
            "turnover": round(float(np.mean([e["mean_turnover"] for e in tacr_expo])), 4),
        }
    )
    vs = pd.DataFrame(summary)
    vs.to_csv(cfg.checkpoint_dir / "vs_model_b.csv", index=False)
    print(vs.round(4).to_string(index=False))

    # wins vs EM: 5-seed protocol
    em = b_rows[b_rows["policy"] == "EM (|a| * m)"].set_index("seed")
    b = b_rows[b_rows["policy"] == "B (naive_new DDR)"].set_index("seed")
    wins_b = int(sum(regime_eval.sharpe_ratio(b.loc[s, "ret"]) > regime_eval.sharpe_ratio(em.loc[s, "ret"]) for s in SEEDS))
    tacr_em = [regime_eval.sharpe_ratio(np.abs(preds_by_seed[s]["action"].to_numpy()) * preds_by_seed[s]["market_ret"].to_numpy()) for s in SEEDS]
    wins_c = int(sum(tacr_sharpe[i] > tacr_em[i] for i in range(5)))
    print(f"\nvs EM (criterion >= 3/5): B wins {wins_b}/5, C (TACR) wins {wins_c}/5")
    print(f"all saved under {cfg.checkpoint_dir}")


if __name__ == "__main__":
    main()