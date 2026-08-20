"""Training entry point for Model B (DDR).

Training signal — direct backprop through the DSR reward
---------------------------------------------------------
The DSR D_t is differentiable w.r.t. the policy action a_t (the strategy
return r_t = a_t * R_{t+1} - cost * |a_t| enters D_t through the EMA
increments), so the policy gradient is obtained by ordinary backprop — no
REINFORCE estimator, no baseline. This is the original paper's "direct
reinforcement" for the trading case and is far more sample-efficient than a
score-function estimate. Chosen over REINFORCE; if the DSR gradient ever
misbehaved, the fallback would be REINFORCE with the same D_t reward.

Offline adaptation (explicit, option (a) of the spec)
-----------------------------------------------------
Original DDR is online/on-policy. Here the policy is trained on the OFFLINE
state sequence (feature windows from Phase 1) while generating its OWN
actions; the DSR reward is computed from those own actions against the
REALIZED next-day market returns in that state. The logged actions and
rewards of the behavior policies are never used. The state sequence is the
market path the agent would have experienced; the action sequence is
entirely the agent's own.

Gradient flow — truncated BPTT
------------------------------
The EMA state (A_t, B_t) carries across the whole train sequence but is
detached at block boundaries (blocks of ``window_size`` consecutive days),
so gradients flow only within each block. D_t is only emitted after
``warmup_steps`` EMA steps (counted once per sequence, so the first block
contributes nothing but EMA seeding).

Outputs (all under ``cfg.checkpoint_dir``, created on demand):
- ``ddr_best.pt``   best state dict by val realized Sharpe (+ config + epoch)
- ``training_log.csv`` per-epoch loss / train EMA Sharpe / val Sharpe

Reward fix (volatility-targeted DSR): with ``cfg.vol_targeting`` (default on
via configs/ddr.yaml) each action is scaled to a target annualized vol of
the strategy-return series BEFORE the DSR increment — see
``VolTargetBuffer`` in dsr.py. Val selection (``roll_sharpe``) and the test
roll always use the RAW actions (the trading P&L); vol-targeting shapes
only the training reward.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.eval.regime_eval import sharpe_ratio
from src.models.ddr.config import DDRConfig
from src.models.ddr.data import DDRData, load_ddr_data, split_ddr_data
from src.models.ddr.dsr import DSRState, VolTargetBuffer
from src.models.ddr.policy import DDRPolicy

BEST_NAME = "ddr_best.pt"


def _strategy_returns(
    actions: torch.Tensor, next_returns: torch.Tensor, cost_bps: float
) -> torch.Tensor:
    return actions * next_returns - (cost_bps / 1e4) * actions.abs()


def _valid(split: DDRData) -> torch.Tensor:
    if split.valid is None:
        return torch.ones(len(split.windows), dtype=torch.bool)
    return split.valid


def roll_sharpe(model: DDRPolicy, split: DDRData, cost_bps: float) -> float:
    """Realized annualized Sharpe of the deterministic policy on a split,
    computed over valid days only (data-hole boundaries excluded)."""
    mask = _valid(split)
    if int(mask.sum()) < 2:
        return float("nan")
    with torch.no_grad():
        actions = model(split.windows[mask]).squeeze(-1)
    returns = _strategy_returns(actions, split.next_returns[mask], cost_bps).numpy()
    return sharpe_ratio(returns, periods_per_year=252)


def _serializable_config(cfg: DDRConfig) -> dict:
    """asdict(cfg) minus non-pickle-safe values (Path -> str), so checkpoints
    load with torch.load(weights_only=True) under torch >= 2.6."""
    return {
        k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()
    }


def train_ddr(
    cfg: DDRConfig, data: DDRData | None = None
) -> tuple[DDRPolicy, pd.DataFrame, Path]:
    """Train the DDR policy; returns (model, history, checkpoint_path)."""
    cfg.validate()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if data is None:
        data = load_ddr_data(cfg.window_size)
    splits = split_ddr_data(data)
    train, val = splits["train"], splits["val"]

    model = DDRPolicy(input_dim=data.windows.shape[2], hidden_size=cfg.hidden_size, rnn_type=cfg.rnn_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val, best_path = float("-inf"), None
    valid_mask = _valid(train)
    train_windows = train.windows[valid_mask]
    train_returns = train.next_returns[valid_mask]
    rows = []
    for epoch in range(cfg.epochs):
        model.train()
        state = DSRState(eta=cfg.eta, warmup_steps=cfg.warmup_steps)
        vt = (
            VolTargetBuffer(
                target_vol=cfg.target_vol,
                window=cfg.vol_target_window,
                max_leverage=cfg.max_leverage,
                cost_bps=cfg.transaction_cost_bps,
            )
            if cfg.vol_targeting
            else None
        )
        losses = []
        block = cfg.window_size
        for i in range(0, len(train_windows), block):
            x = train_windows[i : i + block]
            actions = model(x).squeeze(-1)
            if vt is not None:
                # reward fix: DSR sees vol-targeted returns (scaled_action
                # carries the gradient; the vol denominator is detached)
                _, returns = vt(actions, train_returns[i : i + block])
            else:
                returns = _strategy_returns(
                    actions, train_returns[i : i + block], cfg.transaction_cost_bps
                )
            D = state.update(returns)
            valid = torch.isfinite(D)
            if not valid.any():
                continue
            loss = -D[valid].mean()
            if cfg.exposure_regularization:
                # direct level pressure: force mean|a_t| toward the target on
                # the same steps that contribute to the DSR loss. The DSR
                # gradient is level-free (a uniform de-leverage cancels in
                # the Sharpe ratio); this term restores it explicitly.
                a_pen = actions[valid].abs().mean()
                loss = loss + cfg.exposure_lambda * (a_pen - cfg.target_exposure).abs()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            all_actions = model(train_windows).squeeze(-1)
        train_sharpe_ema = state.sharpe_ema() * (252**0.5)
        val_sharpe = roll_sharpe(model, val, cfg.transaction_cost_bps)
        rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "train_sharpe_ema": float(train_sharpe_ema),
                "val_sharpe_ann": val_sharpe,
                "train_mean_abs_action": float(all_actions.abs().mean().item()),
                "train_short_frac": float((all_actions < 0).float().mean().item()),
            }
        )
        if np.isfinite(val_sharpe) and val_sharpe > best_val:
            best_val = val_sharpe
            best_path = cfg.checkpoint_dir / BEST_NAME
            torch.save(
                {"state_dict": model.state_dict(), "config": _serializable_config(cfg), "epoch": epoch + 1},
                best_path,
            )

    history = pd.DataFrame(rows)
    history.to_csv(cfg.checkpoint_dir / "training_log.csv", index=False)
    if best_path is None:
        best_path = cfg.checkpoint_dir / BEST_NAME
        torch.save({"state_dict": model.state_dict(), "config": _serializable_config(cfg), "epoch": cfg.epochs}, best_path)
    return model, history, best_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Model B (DDR).")
    parser.add_argument("--config", type=str, default=None, help="ddr.yaml path (default configs/ddr.yaml)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--rnn", choices=("gru", "lstm"), default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cost-bps", type=float, default=None)
    parser.add_argument("--no-vol-targeting", action="store_true", help="train naive DSR (raw a_t * ret)")
    parser.add_argument("--target-vol", type=float, default=None)
    parser.add_argument("--max-leverage", type=float, default=None)
    parser.add_argument("--exposure-reg", action="store_true", help="add |mean|a|| - target| level pressure to the loss")
    parser.add_argument("--target-exposure", type=float, default=None)
    parser.add_argument("--exposure-lambda", type=float, default=None)
    parser.add_argument("--tag", type=str, default=None,
                        help="write checkpoints to checkpoints/<tag> instead of the yaml dir")
    args = parser.parse_args(argv)

    cfg = DDRConfig.from_yaml(args.config) if args.config else DDRConfig.from_yaml()
    if args.no_vol_targeting:
        cfg.vol_targeting = False
    overrides = {
        "epochs": args.epochs,
        "hidden_size": args.hidden,
        "rnn_type": args.rnn,
        "eta": args.eta,
        "lr": args.lr,
        "seed": args.seed,
        "transaction_cost_bps": args.cost_bps,
        "target_vol": args.target_vol,
        "max_leverage": args.max_leverage,
        "exposure_regularization": args.exposure_reg or None,
        "target_exposure": args.target_exposure,
        "exposure_lambda": args.exposure_lambda,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.tag is not None:
        cfg.checkpoint_dir = Path(__file__).parent / "checkpoints" / args.tag
    cfg.validate()

    model, history, checkpoint = train_ddr(cfg)
    print(f"trained {cfg.epochs} epochs; best val Sharpe {history['val_sharpe_ann'].max():.3f}")
    print(f"vol_targeting={cfg.vol_targeting}")
    print(history.to_string(index=False))
    print(f"checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()