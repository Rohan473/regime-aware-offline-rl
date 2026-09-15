"""Model C (TACR) offline training — Lee & Moon 2023, bc-regularized actor.

Faithful to the authors' implementation (github.com/VarML/TACR,
tac/training/seq_trainer.py), with the project's adaptations:

Per step (paper Algorithm 1):
  1. sample a minibatch of u-length segments from the behavior-policy
     trajectories (train split only; decision dates within the split),
  2. CRITIC update (offline TD — no environment interaction):
       target_Q = r + gamma * Q_target(s_next, pi_target(s_next))
       critic_loss = MSE(Q(s, a_logged), target_Q)      [gamma 0.99]
  3. ACTOR update (the paper's BC-regularized objective, eq. 4):
       lambda = alpha / |Q|.abs().mean().detach()
       actor_loss = -lambda * Q(s, pi(s)) + MSE(pi(s), a_logged)
     — the behavior-cloning term pins the actor to the logged distribution
       so the critic cannot push it toward unsupported actions.
  4. gradient clip 0.25 (paper), AdamW lr 1e-4 wd 1e-4, warmup 10k steps
     (paper), target networks via polyak tau=0.005 (paper).

Checkpoint selection: every epoch the current actor rolls the VAL split
autoregressively (constant rtg_target, its own actions) and the best-val
Sharpe checkpoint is saved — the Model B protocol, which TACR's paper
lacks (it saves the final model). Basin screening: the best-val epoch is
logged per seed; eval.py applies the >= Model B rule (anomalously late
best-val or < 0.7 cross-seed correlation => suspect basin, reseed).

Entry point: python -m src.models.tacr.train [--seed S] [--config path]
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.regime_eval import sharpe_ratio  # noqa: E402
from src.models.tacr.action_model import ConditionalGaussian, pretrain_action_model  # noqa: E402
from src.models.tacr.config import TACRConfig  # noqa: E402
from src.models.tacr.critic import TACRCritic  # noqa: E402
from src.models.tacr.data import TACRData, load_tacr_data, sample_batch, split_tacr_data  # noqa: E402
from src.models.tacr.eval import roll_actions  # noqa: E402
from src.models.tacr.fixes import bcq_constrain, double_q, par_bc_target  # noqa: E402
from src.models.tacr.policy import TACRPolicy  # noqa: E402

BEST_NAME = "tacr_best.pt"


def _serializable(cfg: TACRConfig) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}


def _build_critics(
    state_dim: int, act_dim: int, n_critics: int, seed: int
) -> list[TACRCritic]:
    """n_critics independent critics. For double Q the two critics MUST
    start from different weights — with an identical init they stay
    identical forever under the same target + optimizer, degenerating the
    min to the single-critic objective."""
    critics = [TACRCritic(state_dim, act_dim) for _ in range(n_critics)]
    if n_critics > 1:
        torch.manual_seed(seed + 997)
        for m in critics[1].modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        torch.manual_seed(seed)
    return critics


def train_tacr(
    cfg: TACRConfig, data: TACRData | None = None
) -> tuple[TACRPolicy, pd.DataFrame, Path]:
    """Train the TACR actor + critic offline; returns (model, history, path)."""
    cfg.validate()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    if data is None:
        data = load_tacr_data(
            cfg.u,
            exclude_policies=cfg.exclude_policies,
            state_macro=cfg.state_macro,
        )
    splits = split_tacr_data(data)
    train, val = splits["train"], splits["val"]

    state_dim = data.states.shape[-1]
    act_dim = 1
    actor = TACRPolicy(
        state_dim=state_dim,
        act_dim=act_dim,
        u=cfg.u,
        embed_dim=cfg.embed_dim,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_inner=cfg.n_inner,
        dropout=cfg.dropout,
        max_ep_len=cfg.max_ep_len,
        action_head=cfg.action_head,
    )
    n_critics = 2 if cfg.use_double_q else 1
    critics = _build_critics(state_dim, act_dim, n_critics, cfg.seed)
    critic_targets = [copy.deepcopy(c) for c in critics]
    critic_opts = [torch.optim.Adam(c.parameters(), lr=cfg.critic_lr) for c in critics]
    actor_target = copy.deepcopy(actor)

    actor_opt = torch.optim.AdamW(
        actor.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        actor_opt, lambda steps: min((steps + 1) / cfg.warmup_steps, 1)
    )

    action_model = None
    if cfg.use_bcq:
        action_model = pretrain_action_model(
            ConditionalGaussian(state_dim, cfg.action_model_hidden),
            train,
            cfg.action_model_steps,
            cfg.action_model_batch,
            cfg.action_model_lr,
            rng,
            cfg.seed,
        )

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val, best_path = float("-inf"), None
    history = []
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        actor.train()
        actor_losses, critic_losses, q_means = [], [], []
        q_spreads, bcq_devs, bcq_sats, par_cos, par_fracs = [], [], [], [], []
        for _ in range(cfg.steps_per_epoch):
            b = sample_batch(train, cfg.u, cfg.batch_size, rng,
                             balanced_families=cfg.balanced_families)
            s, a, r, rtg, ts = b["states"], b["actions"], b["rewards"], b["rtgs"], b["timesteps"]
            B, u = s.shape[:2]
            s_flat = s.reshape(-1, state_dim)
            a_flat = a.reshape(-1)
            r_flat = r.reshape(-1)

            # --- critic update: TD on logged (s, a, r, s') transitions ---
            with torch.no_grad():
                pi_target = actor_target(s, a, rtg, ts)  # (B, u)
                # next-state / next-action per paper: shift by one position,
                # duplicating the last element at the segment end
                next_s = torch.cat([s[:, 1:], s[:, -1:]], dim=1).reshape(-1, state_dim)
                next_a = torch.cat([pi_target[:, 1:], pi_target[:, -1:]], dim=1).reshape(-1)
                if action_model is not None:
                    a_gen_next = action_model(next_s)  # (N, 1) frozen
                    next_a = bcq_constrain(next_a.unsqueeze(-1), a_gen_next, cfg.bcq_phi)
                else:
                    next_a = next_a.unsqueeze(-1)
                if cfg.use_double_q:
                    t = double_q(
                        critic_targets[0](next_s, next_a),
                        critic_targets[1](next_s, next_a),
                        cfg.double_q_mode,
                    )
                else:
                    t = critic_targets[0](next_s, next_a)
                target_Q = (r_flat + cfg.gamma * t).detach()
            step_critic_losses = []
            for ci, copt in zip(critics, critic_opts):
                current_Q = ci(s_flat, a_flat)
                critic_loss = nn.functional.mse_loss(current_Q, target_Q)
                step_critic_losses.append(float(critic_loss.item()))
                copt.zero_grad()
                critic_loss.backward()
                copt.step()

            # --- actor update: -lambda*Q + BC (paper eq. 4), variant-shaped ---
            pi = actor(s, a, rtg, ts)
            pi_flat = pi.reshape(-1)
            if action_model is not None:
                a_gen_s = action_model(s_flat)  # (N, 1) frozen
                a_eval = bcq_constrain(pi_flat.unsqueeze(-1), a_gen_s, cfg.bcq_phi)
            else:
                a_eval = pi_flat.unsqueeze(-1)
            q_list = [c(s_flat, a_eval) for c in critics]
            Q = double_q(q_list[0], q_list[1], cfg.double_q_mode) if cfg.use_double_q else q_list[0]
            lmbda = cfg.alpha / Q.abs().mean().detach()
            q_term = -lmbda * Q.mean()

            use_bc_term = not (action_model is not None and cfg.bcq_bc_coeff == 0.0)
            if not use_bc_term:
                bc = torch.zeros((), dtype=pi.dtype)
                par_cos_mean, par_frac = float("nan"), float("nan")
            elif cfg.use_par:
                # C: Q-gradient direction w.r.t. the actor's RAW output
                g_q = torch.autograd.grad(Q.sum(), pi_flat, retain_graph=True)[0]
                bc_target, cos_v, par_frac = par_bc_target(
                    pi_flat, a_flat, g_q.detach(), cfg.par_phi, cfg.par_cos_thresh
                )
                bc = cfg.par_bc_coeff * nn.functional.mse_loss(
                    pi_flat.unsqueeze(-1), bc_target.detach().unsqueeze(-1)
                )
                par_cos_mean = float(cos_v.mean().item())
            else:
                bc = nn.functional.mse_loss(pi_flat, a_flat)
                par_cos_mean, par_frac = float("nan"), float("nan")
            actor_loss = q_term + bc
            actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
            actor_opt.step()
            scheduler.step()
            global_step += 1

            # --- polyak target updates (paper) ---
            with torch.no_grad():
                for p, tp in zip(actor.parameters(), actor_target.parameters()):
                    tp.data.copy_(cfg.tau * p.data + (1 - cfg.tau) * tp.data)
                for ct, ctp in zip(critics, critic_targets):
                    for p, tp in zip(ct.parameters(), ctp.parameters()):
                        tp.data.copy_(cfg.tau * p.data + (1 - cfg.tau) * tp.data)

            actor_losses.append(float(actor_loss.item()))
            critic_losses.append(float(np.mean(step_critic_losses)))
            q_means.append(float(Q.mean().item()))
            if cfg.use_double_q:
                q_spreads.append(float((q_list[0] - q_list[1]).abs().mean().item()))
            if action_model is not None:
                bcq_devs.append(float((pi_flat.unsqueeze(-1) - a_gen_s).abs().mean().item()))
                bcq_sats.append(
                    float(
                        (pi_flat.unsqueeze(-1) - a_gen_s).abs().gt(cfg.bcq_phi).float().mean().item()
                    )
                )
            if cfg.use_par and use_bc_term:
                par_cos.append(par_cos_mean)
                par_fracs.append(par_frac)

        # --- val roll (constant rtg_target; deterministic policy) ---
        actor.eval()
        val_dates = val.dates[val.valid.numpy()]
        with torch.no_grad():
            a_full = roll_actions(actor, cfg, data, val_dates, cfg.rtg_target,
                                  action_model=action_model, bcq_phi=cfg.bcq_phi)
            a_val = a_full[data.dates.get_indexer(val_dates)]
        m_val = val.market_returns[val.valid.numpy()].numpy()
        val_sharpe = sharpe_ratio(a_val * m_val)

        history.append(
            {
                "epoch": epoch,
                "actor_loss": float(np.mean(actor_losses)),
                "critic_loss": float(np.mean(critic_losses)),
                "q_mean": float(np.mean(q_means)),
                "val_sharpe": val_sharpe,
                "val_mean_abs_action": float(np.abs(a_val).mean()),
                "q1_q2_spread": float(np.mean(q_spreads)) if q_spreads else float("nan"),
                "bcq_mean_dev": float(np.mean(bcq_devs)) if bcq_devs else float("nan"),
                "bcq_sat_frac": float(np.mean(bcq_sats)) if bcq_sats else float("nan"),
                "par_cos": float(np.mean(par_cos)) if par_cos else float("nan"),
                "par_frac_replaced": float(np.mean(par_fracs)) if par_fracs else float("nan"),
            }
        )
        print(
            f"epoch {epoch:02d}: actor {np.mean(actor_losses):.4f} critic "
            f"{np.mean(critic_losses):.4f} Q {np.mean(q_means):.4f} "
            f"val Sharpe {val_sharpe:+.3f} (mean|a| {np.abs(a_val).mean():.3f})",
            flush=True,
        )
        if np.isfinite(val_sharpe) and val_sharpe > best_val:
            best_val = val_sharpe
            best_path = cfg.checkpoint_dir / BEST_NAME
            torch.save(
                _payload(
                    cfg, actor, critics, action_model, state_dim, act_dim,
                    epoch, best_val,
                ),
                best_path,
            )

    hist = pd.DataFrame(history)
    hist.to_csv(cfg.checkpoint_dir / "training_log.csv", index=False)
    torch.save(
        _payload(
            cfg, actor, critics, action_model, state_dim, act_dim,
            cfg.epochs, best_val,
        ),
        cfg.checkpoint_dir / "tacr_final.pt",
    )
    if best_path is None:
        best_path = cfg.checkpoint_dir / BEST_NAME
        torch.save(
            _payload(
                cfg, actor, critics, action_model, state_dim, act_dim,
                cfg.epochs, best_val,
            ),
            best_path,
        )
    print(f"[seed {cfg.seed}] best val Sharpe {best_val:.4f} @ epoch "
          f"{int(hist['val_sharpe'].idxmax()) + 1} -> {best_path}")
    return actor, hist, best_path


def _payload(
    cfg: TACRConfig,
    actor: TACRPolicy,
    critics: list[TACRCritic],
    action_model: ConditionalGaussian | None,
    state_dim: int,
    act_dim: int,
    epoch: int,
    best_val: float,
) -> dict:
    payload = {
        "actor": actor.state_dict(),
        "critic": [c.state_dict() for c in critics],
        "config": _serializable(cfg),
        "state_dim": state_dim,
        "act_dim": act_dim,
        "epoch": epoch,
        "best_val_sharpe": float(best_val),
    }
    if action_model is not None:
        payload["action_model"] = action_model.state_dict()
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Model C (TACR).")
    parser.add_argument("--config", type=str, default=None, help="tacr.yaml path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None,
                        help="warmup steps (paper 10000; scaled to the step budget)")
    parser.add_argument("--u", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None,
                        help="diagnostic: override the paper's critic_lr (1e-6)")
    parser.add_argument("--use-double-q", action="store_true", help="fix A: clipped double Q")
    parser.add_argument("--double-q-mode", choices=("min", "mean"), default=None,
                        help="double-Q aggregation: min (TD3) or mean (ensemble)")
    parser.add_argument("--use-bcq", action="store_true", help="fix B: BCQ-style hard action constraint")
    parser.add_argument("--bcq-phi", type=float, default=None)
    parser.add_argument("--bcq-bc-coeff", type=float, default=None)
    parser.add_argument("--use-par", action="store_true", help="fix C: direction-aware BC target replacement")
    parser.add_argument("--exclude-policies", nargs="*", default=None,
                        help="policy tags to drop from TACR training data (default from yaml: random)")
    parser.add_argument("--no-exclude", action="store_true",
                        help="train on ALL policies incl. random (override yaml exclude_policies)")
    parser.add_argument("--balanced-families", action="store_true",
                        help="family-balanced batch sampler (1/N per family, then window)")
    parser.add_argument("--no-balanced-families", action="store_true",
                        help="use the paper's uniform-over-policies sampler")
    parser.add_argument("--par-phi", type=float, default=None)
    parser.add_argument("--par-cos-thresh", type=float, default=None)
    parser.add_argument("--par-bc-coeff", type=float, default=None)
    parser.add_argument("--state-macro", action="store_true",
                        help="Exp 1 (7.30.1): widen TACR state to 16-dim (8 SPY + "
                             "8 macro causal z), dates restricted to 2007+")
    parser.add_argument("--tag", type=str, default=None,
                        help="append a subdir to checkpoint_dir (diagnostic runs)")
    args = parser.parse_args(argv)

    cfg = TACRConfig.from_yaml(args.config) if args.config else TACRConfig.from_yaml()
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
    if args.u is not None:
        cfg.u = args.u
    if args.n_layer is not None:
        cfg.n_layer = args.n_layer
    if args.alpha is not None:
        cfg.alpha = args.alpha
    if args.critic_lr is not None:
        cfg.critic_lr = args.critic_lr
    if args.use_double_q:
        cfg.use_double_q = True
    if args.double_q_mode is not None:
        cfg.double_q_mode = args.double_q_mode
    if args.use_bcq:
        cfg.use_bcq = True
    if args.bcq_phi is not None:
        cfg.bcq_phi = args.bcq_phi
    if args.bcq_bc_coeff is not None:
        cfg.bcq_bc_coeff = args.bcq_bc_coeff
    if args.use_par:
        cfg.use_par = True
    if args.no_exclude:
        cfg.exclude_policies = ()
    elif args.exclude_policies is not None:
        cfg.exclude_policies = tuple(args.exclude_policies)
    if args.balanced_families:
        cfg.balanced_families = True
    if args.no_balanced_families:
        cfg.balanced_families = False
    if args.par_phi is not None:
        cfg.par_phi = args.par_phi
    if args.par_cos_thresh is not None:
        cfg.par_cos_thresh = args.par_cos_thresh
    if args.par_bc_coeff is not None:
        cfg.par_bc_coeff = args.par_bc_coeff
    if args.state_macro:
        cfg.state_macro = True

    if args.tag is not None:
        cfg.checkpoint_dir = cfg.checkpoint_dir / args.tag
    cfg.checkpoint_dir = cfg.checkpoint_dir / f"s{cfg.seed}"
    train_tacr(cfg)


if __name__ == "__main__":
    main()