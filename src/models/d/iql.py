"""Implicit Q-Learning (IQL) agent for Model D.

Faithful to Kostrikov et al., "Offline Reinforcement Learning with Implicit
Q-Learning" (ICLR 2022; code github.com/ikostrikov/implicit_q_learning).
IQL learns a value function with the EXPECTILE loss — which recovers the
value of the best (upper) tail of the behavior distribution WITHOUT ever
querying Q on out-of-distribution actions — then extracts a policy by
advantage-weighted regression (AWR). This is the Phase-4 objective choice
over Model C's actor-critic (whose Q-collapse failure mode is documented in
PROJECT_NOTES 7.6.1) and over CQL.

Networks (all consume the shared encoder representation h_t):
  - V head:  MLP(h_t) -> scalar
  - Q head:  MLP(concat(h_t, a)) -> scalar
  - policy:  Linear(h_t) -> tanh (deterministic action, same roll protocol
             as Models B/C)

Losses (paper eq. 4-7, with target networks updated by polyak tau):
  - L_V = E[ l2^tau( Q_target(s,a) - V(s) ) ],  l2^tau(u)=|tau - 1(u<0)| u^2
  - L_Q = E[ ( Q(s,a) - (r + gamma*(1-done) * V_target(s')) )^2 ]
  - L_pi = E[ exp(beta * (Q(s,a) - V(s))) * (pi(s) - a)^2 ]   (AWR, MSE
          on the deterministic mean — the paper's continuous-control form;
          the advantage weight is stop-gradient). No separate BC term:
          AWR is implicitly BC-regularized by the exponential weight.

Only the last position of each context window (h_t) is used; the encoder's
causal mask guarantees h_t carries no future information.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.d.backbone import StateEncoder


def expectile_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """l2^tau(u) = |tau - 1(u<0)| * u^2, u = target - pred."""
    diff = (target - pred).detach()
    w = torch.where(diff < 0, 1.0 - tau, tau)
    return (w * (target - pred) ** 2).mean()


class ValueHead(nn.Module):
    def __init__(self, embed_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h).squeeze(-1)


class QHead(nn.Module):
    def __init__(self, embed_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat((h, a.unsqueeze(-1)), dim=-1)
        return self.mlp(x).squeeze(-1)


class IQLAgent(nn.Module):
    """StateEncoder + IQL V/Q/policy heads with target networks.

    forward(s, ts) -> (pi, v, q_actions) for the last position; the train
    loop separately calls value/q/policy on h_t and h_next.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        u: int,
        embed_dim: int = 128,
        n_layer: int = 4,
        n_head: int = 1,
        n_inner: int = 512,
        dropout: float = 0.1,
        max_ep_len: int = 5283,
        tau: float = 0.005,
    ) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.tau = tau
        self.encoder = StateEncoder(
            state_dim=state_dim, u=u, embed_dim=embed_dim, n_layer=n_layer,
            n_head=n_head, n_inner=n_inner, dropout=dropout,
            max_ep_len=max_ep_len,
        )
        self.v_head = ValueHead(embed_dim)
        self.q_head = QHead(embed_dim, act_dim)
        self.policy_head = nn.Sequential(nn.Linear(embed_dim, act_dim))

        self.v_target = copy.deepcopy(self.v_head)
        self.q_target = copy.deepcopy(self.q_head)
        for p in (*self.v_target.parameters(), *self.q_target.parameters()):
            p.requires_grad_(False)

    def reset_targets(self) -> None:
        self.v_target.load_state_dict(self.v_head.state_dict())
        self.q_target.load_state_dict(self.q_head.state_dict())

    def polyak_update(self, tau: float | None = None) -> None:
        t = self.tau if tau is None else tau
        with torch.no_grad():
            for p, tp in zip(self.v_head.parameters(), self.v_target.parameters()):
                tp.data.copy_(t * p.data + (1.0 - t) * tp.data)
            for p, tp in zip(self.q_head.parameters(), self.q_target.parameters()):
                tp.data.copy_(t * p.data + (1.0 - t) * tp.data)

    def encode(self, states: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Full-window encodings (B, u, E)."""
        return self.encoder(states, timesteps)

    def h_last(self, states: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.encoder(states, timesteps)[:, -1]

    def policy(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.policy_head(h)).squeeze(-1)

    def v(self, h: torch.Tensor) -> torch.Tensor:
        return self.v_head(h)

    def q(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.q_head(h, a)

    def v_target_eval(self, h: torch.Tensor) -> torch.Tensor:
        return self.v_target(h)

    def q_target_eval(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.q_target(h, a)

    def forward(self, states: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Deterministic action for the last position (roll protocol)."""
        return self.policy(self.h_last(states, timesteps))