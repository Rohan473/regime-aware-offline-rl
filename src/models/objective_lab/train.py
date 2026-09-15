"""Training loop for all four Objective Lab arms.

Trunk: identical to DDR's block-truncated loop (blocks of ``window``
consecutive decision dates, optimizer step per block, first block starts
the EMA / bootstrapping). Only the loss differs by arm:

  A  MSE(prediction, standardized next-day return)
  B  loss = -D[valid].mean()  (DDR's differential Sharpe ratio)
  C  A2C: REINFORCE on A_t = r_t + gamma*V(h_{t+1}) - V(h_t), + entropy reg
  D  MSE over the MASKED dims of today's z-feature vector, reconstructed
     from the earlier context in the window

Best checkpoint per arm is selected on a VALIDATION metric (Sharpe for the
RL arms, negative MSE for the supervised arms) and stored under
checkpoints/<arm>/s<seed>/best.pt with the same {"state_dict","config"}.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.eval.regime_eval import sharpe_ratio
from src.models.ddr.data import DDRData, load_ddr_data, split_ddr_data
from src.models.ddr.dsr import DSRState, VolTargetBuffer
from src.models.objective_lab.config import tag_path
from src.models.objective_lab.model import build_agent

BEST_NAME = "best.pt"


def _valid_mask(split: DDRData) -> torch.Tensor:
    return split.valid if split.valid is not None else torch.ones(len(split.windows), dtype=torch.bool)


def _strategy_returns(actions, next_returns, cost_bps, prev_action=0.0):
    cost = (cost_bps / 1e4) * torch.abs(actions - prev_action)
    return actions * next_returns - cost


def roll_sharpe(agent, split: DDRData, cost_bps: float) -> float:
    mask = _valid_mask(split)
    if int(mask.sum()) < 2:
        return float("nan")
    with torch.no_grad():
        h = agent.encoder.encode(split.windows[mask])
        a = agent.policy.mu(h)
    r = _strategy_returns(a, split.next_returns[mask], cost_bps).numpy()
    return sharpe_ratio(r, periods_per_year=252)


def _train_sharpe(agent, tw: torch.Tensor, tr: torch.Tensor, cost_bps: float) -> float:
    with torch.no_grad():
        h = agent.encoder.encode(tw)
        a = agent.policy.mu(h)
    r = _strategy_returns(a, tr, cost_bps).numpy()
    return sharpe_ratio(r, periods_per_year=252)


def _pretanh(agent, h_t: torch.Tensor) -> torch.Tensor:
    return agent.policy.head(h_t).squeeze(-1)


def _squashed_logp(pretanh, mu_raw, sigma, a) -> torch.Tensor:
    lp = -0.5 * ((pretanh - mu_raw) / sigma) ** 2
    lp -= torch.log(torch.tensor(sigma)) + 0.5 * torch.log(torch.tensor(2 * 3.141592653589793))
    return lp - torch.log1p(-a.pow(2) + 1e-8)


def _eval_metric(arm: str, agent, val: DDRData, cfg, tstats=None) -> tuple[float, float]:
    """Returns (val metric, train-epoch metric)."""
    if arm in ("B_dsr", "C_actorcritic"):
        return roll_sharpe(agent, val, cfg.cost_bps), 0.0
    mask = _valid_mask(val)
    with torch.no_grad():
        h = agent.encoder.encode(val.windows[mask])
        if arm == "A_predictive":
            pred = agent.head(h)
            y = (val.next_returns[mask] - tstats["mean"]) / tstats["std"]
            vm = -float(torch.mean((pred - y) ** 2).item())
        else:
            pred = agent.head(h)
            vm = -float(torch.mean((pred - val.windows[mask][:, -1, :]) ** 2).item())
    return vm, 0.0


def train_arm(arm: str, cfg, data: DDRData | None = None) -> tuple:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if data is None:
        data = load_ddr_data(cfg.window)
    splits = split_ddr_data(data)
    train, val = splits["train"], splits["val"]
    tmask = _valid_mask(train)
    tw = train.windows[tmask]
    tr = train.next_returns[tmask]

    agent = build_agent(arm, cfg)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr)
    path = tag_path(cfg, arm)
    path.mkdir(parents=True, exist_ok=True)

    tstats = None
    if arm == "A_predictive":
        tstats = {"mean": float(tr.mean()), "std": float(tr.std()) + 1e-8}

    best_val, best_path = float("-inf"), None
    rows = []
    for epoch in range(cfg.epochs):
        agent.train()
        rng = torch.Generator().manual_seed(cfg.seed + epoch)
        state = DSRState(eta=cfg.eta, warmup_steps=cfg.warmup_steps) if arm == "B_dsr" else None
        vt = (
            VolTargetBuffer(target_vol=cfg.target_vol, window=cfg.vol_target_window,
                            max_leverage=cfg.max_leverage, cost_bps=cfg.cost_bps)
            if arm == "B_dsr" and cfg.vol_targeting
            else None
        )
        prev_a = torch.zeros(1)
        losses = []
        for lo in range(0, len(tw), cfg.block):
            hi = min(lo + cfg.block, len(tw))
            x = tw[lo:hi]
            r = tr[lo:hi]
            vmask = tmask[lo:hi]

            if arm == "B_dsr":
                a = agent.policy.mu(agent.encoder.encode(x))
                if vt is not None:
                    _, rs = vt(a, r)
                else:
                    rs = _strategy_returns(a, r, cfg.cost_bps, prev_a)
                    prev_a = a[-1:].detach()
                D = state.update(rs)
                good = vmask & torch.isfinite(D)
                if not good.any():
                    continue
                loss = -D[good].mean()
            elif arm == "C_actorcritic":
                h_all = agent.encoder.encode_all(x)
                h_t = h_all[:, -1, :]                                   # (L, D) decision-day state
                h_next = torch.cat([h_all[1:, -1, :], torch.zeros_like(h_t[:1])], dim=0)  # terminal v=0
                mu_raw = _pretanh(agent, h_t)
                mu = torch.tanh(mu_raw)
                with torch.no_grad():
                    v_next = agent.value(h_next.detach())
                v_t = agent.value(h_t)
                r_t = r * mu
                z = torch.randn_like(mu_raw, generator=rng)
                pretanh = mu_raw + cfg.ac_sigma * z
                a_s = torch.tanh(pretanh)
                logp = _squashed_logp(pretanh, mu_raw, cfg.ac_sigma, a_s)
                td = r_t + cfg.gamma * v_next.detach()
                A = (td - v_t).detach()
                pol = -torch.mean(logp[vmask] * A[vmask]) - cfg.ent_coef * torch.mean(logp[vmask])
                val_l = torch.mean((v_t[vmask] - td[vmask]) ** 2)
                loss = pol + val_l
            elif arm == "A_predictive":
                pred = agent.head(agent.encoder.encode(x))
                y = (r - tstats["mean"]) / tstats["std"]
                loss = torch.mean((pred[vmask] - y[vmask]) ** 2)
            else:  # D_masked
                x_c, target, mask = _mask_last_day(x, cfg.mask_ratio, rng)
                pred = agent.head(agent.encoder.encode(x_c))
                loss = torch.mean((pred[mask] - target[mask]) ** 2)

            if not torch.isfinite(loss).item():
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        agent.eval()
        val_metric, _ = _eval_metric(arm, agent, val, cfg, tstats)
        train_metric = float(np.mean(losses)) if losses else float("nan")
        rows.append({"epoch": epoch + 1, "arm": arm, "train_loss": train_metric,
                     "val_metric": val_metric})
        if np.isfinite(val_metric) and val_metric > best_val:
            best_val = val_metric
            best_path = path / BEST_NAME
            torch.save({"arm": arm, "state_dict": agent.state_dict(),
                        "config": {k: (str(v) if isinstance(v, Path) else v)
                                   for k, v in cfg.__dict__.items()},
                        "epoch": epoch + 1, "val_metric": float(val_metric)}, best_path)
            print(f"[{arm}] epoch {epoch + 1}: val {val_metric:.4f}  (new best)")

    history = pd.DataFrame(rows)
    history.to_csv(path / "training_log.csv", index=False)
    if best_path is None:
        best_path = path / BEST_NAME
        torch.save({"arm": arm, "state_dict": agent.state_dict(), "config": {}, "epoch": cfg.epochs,
                    "val_metric": float("nan")}, best_path)
    return agent, history, best_path


def _mask_last_day(x: torch.Tensor, mask_ratio: float, rng: torch.Generator):
    B, _, F = x.shape
    target = x[:, -1, :].clone()
    mask = torch.rand(B, F, generator=rng) < mask_ratio
    xc = x.clone()
    xc[:, -1, :][mask] = 0.0
    return xc, target, mask


def load_arm(arm: str, cfg, checkpoint: Path):
    agent = build_agent(arm, cfg)
    agent.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
    agent.eval()
    return agent


def extract_arm(arm: str, cfg, checkpoint: Path):
    """Extracted h_t over the full DDR decision-date grid (train+val+test)."""
    from src.interpret.extract import Extracted

    data = load_ddr_data(cfg.window)
    agent = load_arm(arm, cfg, checkpoint)
    with torch.no_grad():
        H = agent.encoder.encode(data.windows).numpy()
    return Extracted(name=f"objlab {arm.split('_', 1)[1]} ({cfg.hidden}d)", dates=data.dates, H=H)