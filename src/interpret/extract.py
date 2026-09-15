"""Hidden-state extraction for the frozen B/C/D checkpoints.

Every extractor returns an :class:`Extracted` — a per-date representation
matrix aligned to its own decision-date grid (the caller aligns against the
shared target grid via ``src.interpret.targets.align``).

- DDR (naive_new): the GRU's final hidden state ``out[:, -1, :]`` (32-d),
  the representation consumed by the policy head before the tanh action.
- TACR (fix_a_nr7, canonical C+ magnitude source): the transformer's
  state-token embedding ``ln_f(blocks(...))[:, 1::3]`` at the LAST position
  (128-d), the representation the action head sees. The context actions are
  the model's own autoregressive roll (constant rtg_target=0.0), matching
  the paper's roll protocol.
- Model D (fuzzy + IQL): the causal encoder's last-position output
  ``encode(windows, ts)[:, -1]`` (128-d), the representation consumed by
  the V/Q/policy heads.

Single-seed (s20260814) for the headline; the paths are explicit so any
other seed tag is a one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.d.config import DConfig
from src.models.d.data import load_d_data, make_windows
from src.models.d.train import load_agent
from src.models.ddr.config import DDRConfig
from src.models.ddr.data import load_ddr_data
from src.models.ddr.policy import DDRPolicy
from src.models.tacr.config import TACRConfig
from src.models.tacr.data import load_tacr_data
from src.models.tacr.eval import load_checkpoint, roll_actions

SEED = 20260814

DDR_CKPT = Path(__file__).resolve().parents[2] / "src/models/ddr/checkpoints/naive_new"
TACR_CKPT = Path(__file__).resolve().parents[2] / "src/models/tacr/checkpoints/tacr/fix_a_nr7"
D_CKPT = Path(__file__).resolve().parents[2] / "src/models/d/checkpoints/d"


@dataclass
class Extracted:
    """Per-date representation matrix from a frozen model."""

    name: str
    dates: pd.DatetimeIndex
    H: np.ndarray      # (N, D)


def _make_windows_1d(values: np.ndarray, ends: np.ndarray, u: int, dtype=torch.float32) -> torch.Tensor:
    """Causal u-day windows over a 1-D time series (left-pad zeros)."""
    ends = np.asarray(ends)
    idx = ends[:, None] - (u - 1) + np.arange(u)[None, :]
    valid = idx >= 0
    idx_c = np.clip(idx, 0, None)
    w = values[idx_c] * valid
    return torch.tensor(w, dtype=dtype)


def extract_ddr(cfg: DDRConfig, checkpoint: Path) -> Extracted:
    """GRU final hidden state for every DDR decision date (train+val+test)."""
    data = load_ddr_data(cfg.window_size)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = DDRPolicy(
        input_dim=data.windows.shape[2],
        hidden_size=cfg.hidden_size,
        rnn_type=cfg.rnn_type,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.no_grad():
        out, _ = model.rnn(data.windows)
    H = out[:, -1, :].numpy()
    return Extracted(name="DDR-GRU (32d)", dates=data.dates, H=H)


def extract_tacr(cfg: TACRConfig, checkpoint: Path) -> Extracted:
    """TACR state-token embedding of the last context position (128-d).

    Context is the model's own autoregressive roll (rtg_target from cfg),
    so the extracted h_t is EXACTLY the representation the deployed action
    head consumed at each decision date.
    """
    data = load_tacr_data(
        cfg.u, exclude_policies=cfg.exclude_policies, state_macro=cfg.state_macro
    )
    model, _, _ = load_checkpoint(checkpoint, cfg)
    pred_actions = roll_actions(model, cfg, data, data.dates, cfg.rtg_target)

    ends = np.arange(len(data.dates))
    states_w = make_windows(data.states, ends, cfg.u)[0]
    actions_w = _make_windows_1d(pred_actions, ends, cfg.u)
    rtgs_w = torch.full_like(actions_w, cfg.rtg_target)
    ts_w = _make_windows_1d(data.timesteps[0].numpy(), ends, cfg.u, dtype=torch.long)

    with torch.no_grad():
        emb = _tacr_state_tokens(model, states_w, actions_w, rtgs_w, ts_w)
    H = emb[:, -1, :].numpy()
    return Extracted(name="TACR transformer (128d)", dates=data.dates, H=H)


def extract_d(cfg: DConfig, checkpoint: Path) -> Extracted:
    """Model D encoder's last-position representation h_t (128-d)."""
    data = load_d_data(cfg)
    agent = load_agent(checkpoint, cfg)
    ends = np.arange(len(data.dates))
    windows, ts = make_windows(data.states_in, ends, cfg.u)
    with torch.no_grad():
        H = agent.encode(windows, ts)[:, -1].numpy()
    return Extracted(name="D encoder (128d)", dates=data.dates, H=H)


def _tacr_state_tokens(
    model, states: torch.Tensor, actions: torch.Tensor, rtgs: torch.Tensor, timesteps: torch.Tensor
) -> torch.Tensor:
    """State-token embeddings (B, u, embed) — mirrors TACRPolicy.forward."""
    B, u = states.shape[:2]
    s_emb = model.embed_state(states) + model.embed_timestep(timesteps)
    a_emb = model.embed_action(actions.unsqueeze(-1)) + model.embed_timestep(timesteps)
    r_emb = model.embed_return(rtgs.unsqueeze(-1)) + model.embed_timestep(timesteps)
    stacked = torch.stack((r_emb, s_emb, a_emb), dim=2).reshape(B, 3 * u, model.embed_dim)
    stacked = model.embed_ln(stacked)
    x = model.blocks(stacked)
    x = model.ln_f(x)
    return x[:, 1::3]


def tacr_action_from_h(model, h: torch.Tensor) -> np.ndarray:
    """tanh(action head) over last-position embeddings h (N, E) -> (N,).
    Handles TACR (predict_action) and DDR (head) policy heads."""
    head = getattr(model, "predict_action", getattr(model, "head", None))
    with torch.no_grad():
        return torch.tanh(head(h)).squeeze(-1).numpy()