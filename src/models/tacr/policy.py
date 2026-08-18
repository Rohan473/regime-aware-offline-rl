"""TACR actor: GPT-2-style causal transformer over MDP-element triples.

Reproduces the paper's TransformerActor (github.com/VarML/TACR,
tac/models/transformer_actor.py) on our continuous action space:

- separate linear embeddings per modality — return-to-go, state, action —
  plus a learned timestep embedding added to each (the paper's GPT-2 config
  has positional embeddings REMOVED; the timestep embedding plays that
  role), summed per Decision-Transformer convention (paper eq. 1),
- tokens stacked as interleaved triples (r_t, s_t, a_t, r_{t+1}, ...) so the
  sequence is 3*u tokens long,
- L decoder blocks with STRICT CAUSAL self-attention: position k attends
  only to tokens < k (a non-causal mask would leak future information and
  invalidate every downstream result),
- action head: Linear + tanh (adaptation — the paper's Linear+Softmax head
  targets discrete portfolio weights; our actions are continuous in [-1, 1]).
  The tanh head keeps the policy deterministic at roll time, matching the
  deterministic-roll protocol of Model B.

The paper also defines state/return prediction heads ("we don't predict
states or returns for the paper" — unused); they are omitted here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Single-head causal self-attention with an explicit lower-triangular
    mask (paper's n_head=1 setup; multi-head is the same mask broadcast)."""

    def __init__(self, embed_dim: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = embed_dim // n_head
        assert self.head_dim * n_head == embed_dim
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # causal: position i attends to j <= i (strictly lower-triangular incl. diagonal)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)).view(1, 1, T, T)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        y = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(y)


class DecoderBlock(nn.Module):
    """Pre-LN decoder block (GPT-2 convention): causal attention + MLP."""

    def __init__(self, embed_dim: int, n_head: int, n_inner: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_head, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, n_inner),
            nn.ReLU(),
            nn.Linear(n_inner, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TACRPolicy(nn.Module):
    """Decision-Transformer-style actor over (rtg, state, action) triples."""

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
        action_head: str = "tanh",
    ) -> None:
        super().__init__()
        self.u = u
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.action_head = action_head

        # per-modality embeddings + learned timestep embedding (paper eq. 1)
        self.embed_return = nn.Linear(1, embed_dim)
        self.embed_state = nn.Linear(state_dim, embed_dim)
        self.embed_action = nn.Linear(act_dim, embed_dim)
        self.embed_timestep = nn.Embedding(max_ep_len, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)

        blocks = [
            DecoderBlock(embed_dim, n_head, n_inner, dropout) for _ in range(n_layer)
        ]
        self.blocks = nn.Sequential(*blocks)
        self.ln_f = nn.LayerNorm(embed_dim)

        # action head: Linear + tanh (paper: Linear + Softmax for discrete
        # portfolio weights; continuous [-1, 1] adaptation, documented)
        self.predict_action = nn.Linear(embed_dim, act_dim)

        nn.init.normal_(self.embed_timestep.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.embed_return.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.embed_state.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.embed_action.weight, mean=0.0, std=0.02)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the action for every position from the causal context.

        states (B, u, F); actions (B, u); rtgs (B, u); timesteps (B, u) long.
        Returns action predictions (B, u) — position t predicts a_t from
        (rtg_t, s_t, a_<t) with strict causality: the token of a_t itself is
        NOT attended (it follows the state token in the triple order).
        """
        B, u = states.shape[:2]
        s_emb = self.embed_state(states) + self.embed_timestep(timesteps)
        a_emb = self.embed_action(actions.unsqueeze(-1)) + self.embed_timestep(timesteps)
        r_emb = self.embed_return(rtgs.unsqueeze(-1)) + self.embed_timestep(timesteps)

        # interleave as (r_1, s_1, a_1, r_2, s_2, ...) — 3u tokens
        stacked = torch.stack((r_emb, s_emb, a_emb), dim=2).reshape(B, 3 * u, self.embed_dim)
        stacked = self.embed_ln(stacked)

        x = stacked
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # state tokens sit at positions 3t+1; predict a_t from s_t's context
        state_tokens = x[:, 1::3]
        preds = self.predict_action(state_tokens)
        if self.action_head == "tanh":
            preds = torch.tanh(preds)
        return preds.squeeze(-1) if self.act_dim == 1 else preds