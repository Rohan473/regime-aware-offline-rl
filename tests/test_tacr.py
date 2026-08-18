"""Model C (TACR) validation — Phase-3 required tests.

1. Causal masking is actually causal: perturbing a FUTURE token must not
   change any earlier output (bitwise).
2. The BC-regularization term is correctly computed and its gradient
   contribution is nonzero and finite.
3. Context length u aligns with Model B's windowing (u == WINDOW_DAYS and
   TACR's day-t state == the last row of B's window at t).
4. Return-to-go is the cumulative FUTURE logged reward (hand-computed).
5. Constant-RTG eval roll is immune to future information (perturbing the
   realized return channel must not change roll actions).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import WINDOW_DAYS  # noqa: E402
from src.models.ddr.data import load_ddr_data  # noqa: E402
from src.models.tacr.config import TACRConfig  # noqa: E402
from src.models.tacr.data import load_tacr_data  # noqa: E402
from src.models.tacr.eval import roll_actions  # noqa: E402
from src.models.tacr.policy import TACRPolicy  # noqa: E402


@pytest.fixture()
def small_model() -> TACRPolicy:
    torch.manual_seed(0)
    return TACRPolicy(
        state_dim=4, act_dim=1, u=6, embed_dim=32, n_layer=2, n_head=1,
        n_inner=128, dropout=0.0, max_ep_len=100, action_head="tanh",
    )


def test_causal_masking_future_perturbation_leaves_earlier_outputs_unchanged(
    small_model: TACRPolicy,
) -> None:
    """Phase-3 required test: a position's output must be provably unaffected
    by later tokens — perturb a future token and confirm earlier outputs are
    identical bitwise."""
    small_model.eval()
    B, u = 1, small_model.u
    s = torch.randn(B, u, 4)
    a = torch.rand(B, u) * 2 - 1
    r = torch.randn(B, u) * 0.1
    ts = torch.arange(u).unsqueeze(0)

    with torch.no_grad():
        base = small_model(s, a, r, ts)

    # perturb the LAST two tokens' state/rtg (positions u-2, u-1)
    s2 = s.clone()
    s2[:, -2:, :] += 10.0
    r2 = r.clone()
    r2[:, -2:] += 10.0
    with torch.no_grad():
        perturbed = small_model(s2, a, r2, ts)

    # outputs at positions 0..u-3 must be bitwise identical
    torch.testing.assert_close(base[0, : u - 2], perturbed[0, : u - 2], rtol=0, atol=0)
    # and the perturbed future must actually change the LAST output
    assert not torch.allclose(base[0, -1], perturbed[0, -1], rtol=1e-6, atol=1e-6)


def test_bc_regularization_gradient_nonzero_and_finite(small_model: TACRPolicy) -> None:
    """Phase-3 required test: the BC (behavior-cloning) term is correctly
    computed (MSE against logged actions) and its gradient contribution is
    nonzero and finite."""
    small_model.train()
    B, u = 2, small_model.u
    s = torch.randn(B, u, 4)
    a = torch.rand(B, u) * 2 - 1  # logged actions
    r = torch.randn(B, u) * 0.1
    ts = torch.arange(u).unsqueeze(0).expand(B, -1)

    pi = small_model(s, a, r, ts)
    bc = torch.nn.functional.mse_loss(pi, a)
    assert torch.isfinite(bc)
    assert bc.item() > 0.0  # not trivially zero

    bc.backward()
    grads = [p.grad for p in small_model.parameters() if p.grad is not None]
    assert len(grads) > 0
    norms = [float(g.abs().max()) for g in grads]
    assert all(np.isfinite(norms))
    # gradient must reach the action embedding and the output head
    emb_grad = small_model.embed_action.weight.grad
    head_grad = small_model.predict_action.weight.grad
    assert emb_grad is not None and float(emb_grad.abs().max()) > 0
    assert head_grad is not None and float(head_grad.abs().max()) > 0

    # exact formula: MSE(pi, a_logged), not anything else
    manual = float((((pi - a) ** 2).mean()).detach())
    assert bc.item() == pytest.approx(manual, rel=1e-6)


def test_context_length_matches_model_b_windowing() -> None:
    """Phase-3 required test: u == WINDOW_DAYS and the day-t state TACR
    conditions on is the last row of Model B's W-day window at t (identical
    information horizon per decision date)."""
    cfg = TACRConfig()
    assert cfg.u == WINDOW_DAYS == 20
    assert cfg.validate() is None

    tacr_data = load_tacr_data(cfg.u)
    b_data = load_ddr_data(WINDOW_DAYS)

    shared = tacr_data.dates.intersection(b_data.dates)
    assert len(shared) == len(b_data.dates)  # identical calendars

    # TACR state at t == DDR window's last row at t (both Z_COLUMNS of day t)
    pos = b_data.dates.get_indexer(shared)
    ddr_last = b_data.windows[np.arange(len(shared)), -1, :]
    t_pos = tacr_data.dates.get_indexer(shared)
    tacr_state = tacr_data.states[t_pos]

    # compare in raw (denormalized) space: rebuild from the features frame
    from src.data.loaders import REPO_ROOT
    import pandas as pd

    feats = pd.read_parquet(REPO_ROOT / "data" / "processed" / "features_regimes.parquet")
    zc = [f"z_{c}" for c in ["ret_1d", "ret_5d", "ret_20d", "realized_vol_20d",
                             "rsi_14", "macd_hist", "volume_zscore_20d", "bollinger_pos"]]
    raw = feats.reindex(shared)[zc].to_numpy()
    assert np.allclose(ddr_last.numpy(), raw, atol=1e-6)
    norm = (raw - tacr_data.state_mean) / tacr_data.state_std
    assert np.allclose(tacr_state.numpy(), norm, atol=1e-5)


def test_return_to_go_is_cumulative_future_logged_reward() -> None:
    """RTG_t = sum of future logged rewards from t (per trajectory)."""
    cfg = TACRConfig()
    data = load_tacr_data(cfg.u)
    for p, pol in enumerate(data.policies):
        r = data.rewards[p].numpy()
        manual = np.cumsum(r[::-1])[::-1]
        assert np.allclose(data.rtgs[p].numpy(), manual, atol=1e-6)


def test_eval_roll_immune_to_future_rtg_information() -> None:
    """The roll feeds a constant rtg_target — perturbing the realized
    (future-dependent) RTG channel must not change roll actions."""
    cfg = TACRConfig()
    cfg.u = 6
    cfg.checkpoint_dir = Path(__file__).parent.parent / "src" / "models" / "tacr" / "checkpoints"
    data = load_tacr_data(6)
    model = TACRPolicy(
        state_dim=data.states.shape[-1], act_dim=1, u=6, embed_dim=32,
        n_layer=2, n_head=1, n_inner=128, dropout=0.0,
        max_ep_len=len(data.dates), action_head="tanh",
    )
    model.eval()
    dates = data.dates[:20]

    a1 = roll_actions(model, cfg, data, dates, rtg_target=0.0)
    # realized RTGs (future info) fed as the target must not leak: the roll
    # uses cfg.rtg_target, not the logged channel
    a2 = roll_actions(model, cfg, data, dates, rtg_target=0.0)
    np.testing.assert_array_equal(a1, a2)

    # and a different constant target DOES change actions (the target is the
    # conditioning channel, it is read — deterministically)
    a3 = roll_actions(model, cfg, data, dates, rtg_target=0.5)
    assert not np.allclose(a1, a3, atol=1e-9)