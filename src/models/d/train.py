"""Model D offline training — IQL (Kostrikov et al. 2022) over transitions.

Per step (paper Algorithm 1, adapted to a shared causal-encoder backbone):
  1. sample batch_size transitions (s_t, a, r, s', done) from the TRAIN
     split (valid next-day transitions only),
  2. ONE encoder forward over the 2B windows (s_t and s' contexts) -> h_t,
     h_{t+1},
  3. VALUE update: expectile loss on V(h_t) vs Q_target(h_t, a)   [tau 0.7],
  4. Q update:      MSE(Q(h_t, a), r + gamma (1-done) V_target(h_{t+1})),
  5. POLICY update: exp(beta (Q - V))-weighted MSE of the deterministic
     mean against the logged action (AWR; no separate BC term),
  6. polyak target updates (tau 0.005), AdamW lr 3e-4, no weight decay, no
     gradient clip (IQL paper), linear warmup 1000 steps (CPU-scaled).

Checkpoint selection: every epoch the policy rolls the VAL split (one
batched forward; deterministic tanh actions) and the best-val Sharpe
checkpoint is saved alongside the FINAL checkpoint — the standing
final-epoch sanity-check protocol from PROJECT_NOTES 7.6 (Model C's
val-best selection failed to generalize, so D always evaluates BOTH).

Entry point: python -m src.models.d.train --variant d|d_minus_fuzzy
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.regime_eval import sharpe_ratio  # noqa: E402
from src.models.d.config import DConfig  # noqa: E402
from src.models.d.data import DData, build_batch, load_d_data, make_windows, split_d_data  # noqa: E402
from src.models.d.iql import IQLAgent, expectile_loss  # noqa: E402

BEST_NAME = "d_best.pt"
FINAL_NAME = "d_final.pt"
VARIANT_DIRS = {"d": "d", "d_minus_fuzzy": "d_minus_fuzzy"}
ADV_CLAMP = 10.0  # numerical safety on exp(beta*adv); inactive at daily-return scale


def _serializable(cfg: DConfig) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}


def roll_split(
    agent: IQLAgent, cfg: DConfig, data_full: DData, split: DData
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic policy roll over a split's valid decision dates.

    Batched: one encoder forward over all windows. Returns (dates, actions,
    market_returns) aligned to the valid dates.
    """
    valid_mask = split.valid.numpy()
    ends = split.global_pos[valid_mask]
    windows, ts = make_windows(data_full.states_in, ends, cfg.u)
    with torch.no_grad():
        a = agent.policy(agent.h_last(windows, ts)).numpy()
    m = split.market_returns[valid_mask].numpy()
    dates = split.dates[valid_mask].to_numpy()
    return dates, a, m


def train_d(
    cfg: DConfig, data_full: DData | None = None
) -> tuple[IQLAgent, pd.DataFrame, Path]:
    """Train the IQL agent offline; returns (agent, history, best_path)."""
    cfg.validate()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    if data_full is None:
        data_full = load_d_data(cfg)
    splits = split_d_data(data_full)
    train, val = splits["train"], splits["val"]

    state_in_dim = data_full.states_in.shape[-1]
    assert state_in_dim == cfg.state_in_dim, (
        f"data states_in {state_in_dim} != cfg.state_in_dim {cfg.state_in_dim} "
        f"(fuzzy={cfg.fuzzy})"
    )
    agent = IQLAgent(
        state_dim=state_in_dim,
        act_dim=1,
        u=cfg.u,
        embed_dim=cfg.embed_dim,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_inner=cfg.n_inner,
        dropout=cfg.dropout,
        max_ep_len=cfg.max_ep_len,
        tau=cfg.tau,
    )

    enc_params = list(agent.encoder.parameters())
    v_opt = torch.optim.AdamW(enc_params + list(agent.v_head.parameters()),
                              lr=cfg.lr, weight_decay=cfg.weight_decay)
    q_opt = torch.optim.AdamW(enc_params + list(agent.q_head.parameters()),
                              lr=cfg.lr, weight_decay=cfg.weight_decay)
    pi_opt = torch.optim.AdamW(enc_params + list(agent.policy_head.parameters()),
                               lr=cfg.lr, weight_decay=cfg.weight_decay)

    def warmup(opt):
        return torch.optim.lr_scheduler.LambdaLR(
            opt, lambda steps: min((steps + 1) / cfg.warmup_steps, 1.0)
        )

    scheds = [warmup(o) for o in (v_opt, q_opt, pi_opt)]

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val, best_path = float("-inf"), None
    history = []

    for epoch in range(1, cfg.epochs + 1):
        agent.train()
        v_losses, q_losses, pi_losses, adv_means = [], [], [], []
        for _ in range(cfg.steps_per_epoch):
            b = build_batch(train, data_full, cfg, rng)
            st, tst = b["states_t"], b["timesteps_t"]
            sn, tsn = b["states_next"], b["timesteps_next"]
            a, r, done = b["actions"], b["rewards"], b["dones"]

            # one encoder forward over the 2B windows (s_t + s' contexts)
            both_s = torch.cat((st, sn), dim=0)
            both_ts = torch.cat((tst, tsn), dim=0)
            both_h = agent.encode(both_s, both_ts)
            h_t, h_next = both_h[: st.shape[0]], both_h[st.shape[0]:]
            h_t = h_t[:, -1]
            h_next = h_next[:, -1]

            # --- value: expectile regression of V(s) on Q_target(s, a) ---
            with torch.no_grad():
                q_target = agent.q_target_eval(h_t, a)
            v_pred = agent.v(h_t)
            v_loss = expectile_loss(v_pred, q_target, cfg.expectile)

            # --- Q: TD bootstrap on V_target(s') ---
            with torch.no_grad():
                v_next = agent.v_target_eval(h_next)
                boot = r + cfg.gamma * (~done).float() * v_next
            q_pred = agent.q(h_t, a)
            q_loss = F.mse_loss(q_pred, boot)

            # --- policy: AWR (exp(beta*(Q-V))-weighted MSE, stop-grad weight) ---
            with torch.no_grad():
                adv = (q_pred - v_pred).clamp(max=ADV_CLAMP)
                weights = torch.exp(cfg.temperature * adv)
            pi = agent.policy(h_t)
            pi_loss = (weights * (pi - a) ** 2).mean()

            for opt in (v_opt, q_opt, pi_opt):
                opt.zero_grad()
            # single backward over the summed loss: each optimizer steps the
            # params it owns (shared encoder + its own head), which is
            # mathematically identical to three separate backward/step calls
            # (retain_graph) — the shared encoder receives the sum of the
            # three gradients, exactly the paper's per-network updates.
            (v_loss + q_loss + pi_loss).backward()
            for opt in (v_opt, q_opt, pi_opt):
                opt.step()
            for sched in scheds:
                sched.step()
            agent.polyak_update()

            v_losses.append(float(v_loss.item()))
            q_losses.append(float(q_loss.item()))
            pi_losses.append(float(pi_loss.item()))
            adv_means.append(float(adv.mean().item()))

        # --- val roll (deterministic policy, batched) ---
        agent.eval()
        _, a_val, m_val = roll_split(agent, cfg, data_full, val)
        val_sharpe = sharpe_ratio(a_val * m_val)

        history.append(
            {
                "epoch": epoch,
                "v_loss": float(np.mean(v_losses)),
                "q_loss": float(np.mean(q_losses)),
                "pi_loss": float(np.mean(pi_losses)),
                "adv_mean": float(np.mean(adv_means)),
                "val_sharpe": val_sharpe,
                "val_mean_abs_action": float(np.abs(a_val).mean()),
            }
        )
        print(
            f"epoch {epoch:02d}: v {np.mean(v_losses):.5f} q {np.mean(q_losses):.5f} "
            f"pi {np.mean(pi_losses):.5f} adv {np.mean(adv_means):.4f} "
            f"val Sharpe {val_sharpe:+.3f} (mean|a| {np.abs(a_val).mean():.3f})",
            flush=True,
        )
        if np.isfinite(val_sharpe) and val_sharpe > best_val:
            best_val = val_sharpe
            best_path = cfg.checkpoint_dir / BEST_NAME
            torch.save(
                {
                    "agent": agent.state_dict(),
                    "config": _serializable(cfg),
                    "state_in_dim": state_in_dim,
                    "act_dim": 1,
                    "epoch": epoch,
                    "best_val_sharpe": float(best_val),
                },
                best_path,
            )

    hist = pd.DataFrame(history)
    hist.to_csv(cfg.checkpoint_dir / "training_log.csv", index=False)
    torch.save(
        {
            "agent": agent.state_dict(),
            "config": _serializable(cfg),
            "state_in_dim": state_in_dim,
            "act_dim": 1,
            "epoch": cfg.epochs,
            "best_val_sharpe": float(best_val),
        },
        cfg.checkpoint_dir / FINAL_NAME,
    )
    if best_path is None:
        best_path = cfg.checkpoint_dir / BEST_NAME
        torch.save(
            {
                "agent": agent.state_dict(),
                "config": _serializable(cfg),
                "state_in_dim": state_in_dim,
                "act_dim": 1,
                "epoch": cfg.epochs,
                "best_val_sharpe": float(best_val),
            },
            best_path,
        )
    print(
        f"[seed {cfg.seed}] best val Sharpe {best_val:.4f} @ epoch "
        f"{int(hist['val_sharpe'].idxmax()) + 1} -> {best_path}"
    )
    return agent, hist, best_path


def load_agent(checkpoint: Path, cfg: DConfig) -> IQLAgent:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent = IQLAgent(
        state_dim=payload["state_in_dim"],
        act_dim=payload["act_dim"],
        u=cfg.u,
        embed_dim=cfg.embed_dim,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_inner=cfg.n_inner,
        dropout=cfg.dropout,
        max_ep_len=cfg.max_ep_len,
        tau=cfg.tau,
    )
    agent.load_state_dict(payload["agent"])
    agent.eval()
    return agent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Model D (IQL).")
    parser.add_argument("--config", type=str, default=None, help="model_d.yaml path")
    parser.add_argument(
        "--variant", type=str, default="d", choices=("d", "d_minus_fuzzy"),
        help="d = fuzzy (26-d); d_minus_fuzzy = ablation (8-d, identical net)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--expectile", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--tag", type=str, default=None,
                        help="append a subdir to checkpoint_dir (diagnostic runs)")
    args = parser.parse_args(argv)

    cfg = DConfig.from_yaml(args.config) if args.config else DConfig.from_yaml()
    cfg.fuzzy = args.variant == "d"
    if args.seed is not None:
        cfg.seed = args.seed
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.steps_per_epoch is not None:
        cfg.steps_per_epoch = args.steps_per_epoch
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.warmup is not None:
        cfg.warmup_steps = args.warmup
    if args.expectile is not None:
        cfg.expectile = args.expectile
    if args.temperature is not None:
        cfg.temperature = args.temperature

    if args.tag is not None:
        cfg.checkpoint_dir = cfg.checkpoint_dir / args.tag
    cfg.checkpoint_dir = cfg.checkpoint_dir / VARIANT_DIRS[args.variant] / f"s{cfg.seed}"
    train_d(cfg)


if __name__ == "__main__":
    main()