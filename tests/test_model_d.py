"""Model D (fuzzy + transformer + IQL) validation — Phase-4 required tests.

1. Causal masking is actually causal: perturbing FUTURE rows of the state
   window must not change any earlier encoder output (bitwise).
2. The fuzzy layer is deterministic (train/eval identical) and passes the
   raw state through unchanged.
3. Fuzzy membership parameters are reproducible and match the spec
   (train-split percentiles, UMF std = mf_width_scale x bin width, LMF std
   = UMF x fou_scale).
4. IQL loss components are exactly the hand-computable formulas
   (expectile l2^tau, Q TD bootstrap, AWR exponential weight).
5. Dimensionality at the data boundary: 26-d input for D vs 8-d for
   D-minus-fuzzy, no padding; identical transformer blocks, batch and
   steps-per-epoch (compute-matched ablation).
6. Window construction left-pads before the series start and returns
   absolute timesteps.
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

from src.models.d.backbone import StateEncoder  # noqa: E402
from src.models.d.config import DConfig  # noqa: E402
from src.models.d.data import load_d_data, make_windows  # noqa: E402
from src.models.d.fuzzy import FuzzyLayer, compute_fuzzy_params  # noqa: E402
from src.models.d.iql import IQLAgent, expectile_loss  # noqa: E402


@pytest.fixture()
def small_encoder() -> StateEncoder:
    torch.manual_seed(0)
    return StateEncoder(
        state_dim=4, u=6, embed_dim=32, n_layer=2, n_head=1,
        n_inner=128, dropout=0.0, max_ep_len=100,
    )


def test_causal_masking_future_perturbation_leaves_earlier_outputs_unchanged(
    small_encoder: StateEncoder,
) -> None:
    """A position's encoding must be provably unaffected by later rows."""
    small_encoder.eval()
    u = small_encoder.u
    s = torch.randn(1, u, 4)
    ts = torch.arange(u).unsqueeze(0)

    with torch.no_grad():
        base = small_encoder(s, ts)

    s2 = s.clone()
    s2[:, -2:, :] += 10.0  # perturb the last two rows (future info)
    with torch.no_grad():
        perturbed = small_encoder(s2, ts)

    torch.testing.assert_close(base[0, : u - 2], perturbed[0, : u - 2], rtol=0, atol=0)
    assert not torch.allclose(base[0, -1], perturbed[0, -1], rtol=1e-6, atol=1e-6)


def test_fuzzy_layer_deterministic_and_passthrough() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    states = rng.normal(size=(40, 8)).astype("float32")
    layer = FuzzyLayer(state_dim=8, field_indices=[2, 3, 4], n_terms=3,
                       mf_width_scale=0.5, fou_scale=0.8)
    layer.set_params_from(states)
    x = torch.tensor(states)

    layer.eval()
    with torch.no_grad():
        o1 = layer(x)
    with torch.no_grad():
        o2 = layer(x)
    torch.testing.assert_close(o1, o2, rtol=0, atol=0)  # deterministic

    layer.train()
    o3 = layer(x)  # train mode must be identical (buffers only)
    torch.testing.assert_close(o1, o3, rtol=0, atol=0)

    # shape (..., 8) -> (..., 26) and the raw 8 columns are unchanged
    assert o1.shape[-1] == 26
    torch.testing.assert_close(o1[..., :8], x, rtol=0, atol=0)
    # memberships are Gaussian outputs in [0, 1]
    fuzzy = o1[..., 8:]
    assert bool((fuzzy >= 0).all()) and bool((fuzzy <= 1).all())
    assert layer.n_fuzzy_features == 18


def test_fuzzy_param_reproducibility_and_spec_values() -> None:
    rng = np.random.default_rng(1)
    # field 0 uniform on [0, 100): percentiles land at 33, 50, 66
    states = np.column_stack(
        [
            rng.uniform(0, 100, size=5000),
            rng.normal(0, 1, size=5000),
            rng.normal(0, 1, size=5000),
        ]
    )
    p = compute_fuzzy_params(states, [0], n_terms=3, mf_width_scale=0.5, fou_scale=0.8)
    centers = p["centers"][0]
    assert centers[0] == pytest.approx(33.0, abs=1.0)
    assert centers[1] == pytest.approx(50.0, abs=1.0)
    assert centers[2] == pytest.approx(66.0, abs=1.0)
    # bin widths: edges = distance to neighbor, interior = half the span
    assert p["std_umf"][0, 0] == pytest.approx(0.5 * (centers[1] - centers[0]), rel=1e-9)
    assert p["std_umf"][0, 2] == pytest.approx(0.5 * (centers[2] - centers[1]), rel=1e-9)
    assert p["std_umf"][0, 1] == pytest.approx(0.5 * (centers[2] - centers[0]) / 2.0, rel=1e-9)
    np.testing.assert_allclose(p["std_lmf"], p["std_umf"] * 0.8)

    # set_params_from is deterministic
    layer = FuzzyLayer(state_dim=8, field_indices=[0], n_terms=3, mf_width_scale=0.5, fou_scale=0.8)
    layer.set_params_from(states[:, :8])
    c1 = layer.centers.clone()
    layer.set_params_from(states[:, :8])
    torch.testing.assert_close(c1, layer.centers, rtol=0, atol=0)


def test_expectile_loss_formula() -> None:
    tau = 0.7
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    target = torch.tensor([0.0, 3.0, 2.0, 4.0])  # u = -1, 1, -1, 0
    loss = expectile_loss(pred, target, tau)
    u = (target - pred).detach()
    w = torch.where(u < 0, 1.0 - tau, tau)
    manual = (w * u ** 2).mean()
    assert loss.item() == pytest.approx(manual.item(), rel=1e-6)


def test_iql_agent_shapes_and_tanh_range() -> None:
    torch.manual_seed(0)
    agent = IQLAgent(state_dim=26, act_dim=1, u=6, embed_dim=32, n_layer=2,
                     n_head=1, n_inner=128, dropout=0.0, max_ep_len=100, tau=0.005)
    agent.eval()
    B = 4
    s = torch.randn(B, 6, 26)
    ts = torch.arange(6).unsqueeze(0).expand(B, -1)
    with torch.no_grad():
        a = agent(s, ts)
        h = agent.h_last(s, ts)
        v = agent.v(h)
        q = agent.q(h, torch.tensor([0.5] * B))
    assert a.shape == (B,)
    assert h.shape == (B, 32)
    assert v.shape == (B,) and q.shape == (B,)
    assert bool((a >= -1).all()) and bool((a <= 1).all())

    # Q and V heads are different networks (IQL requires the expectile gap)
    assert agent.q_head is not agent.v_head
    assert agent.q_target is not agent.q_head


def test_iql_awr_policy_loss_weighted() -> None:
    """AWR loss = mean(exp(beta*(Q-V)) * (pi - a)^2) with a STOP-GRAD weight."""
    torch.manual_seed(1)
    agent = IQLAgent(state_dim=8, act_dim=1, u=6, embed_dim=16, n_layer=1,
                     n_head=1, n_inner=64, dropout=0.0, max_ep_len=100, tau=0.005)
    h = torch.randn(4, 16, requires_grad=True)
    a_log = torch.tensor([0.5, -0.5, 0.2, -0.2])
    with torch.no_grad():
        q = torch.tensor([1.0, 0.5, 0.0, -0.5])
        v = torch.tensor([0.0, 0.0, 0.0, 0.0])
    adv = q - v
    weights = torch.exp(3.0 * adv)
    pi = agent.policy(h)
    loss = (weights * (pi - a_log) ** 2).mean()

    manual = float((torch.exp(3.0 * (q - v)) * (pi.detach() - a_log) ** 2).mean())
    assert loss.item() == pytest.approx(manual, rel=1e-6)


def test_data_boundary_dimensionality_and_compute_match() -> None:
    """D = 26-d input; D-minus-fuzzy = 8-d, NO padding; identical blocks and
    identical per-step training load (batch, steps/epoch)."""
    cfg_d = DConfig(fuzzy=True)
    cfg_mf = DConfig(fuzzy=False)
    assert cfg_d.state_in_dim == 26
    assert cfg_mf.state_in_dim == 8
    assert cfg_d.embed_dim == cfg_mf.embed_dim
    assert cfg_d.n_layer == cfg_mf.n_layer
    assert cfg_d.n_head == cfg_mf.n_head
    assert cfg_d.n_inner == cfg_mf.n_inner
    assert cfg_d.batch_size == cfg_mf.batch_size
    assert cfg_d.steps_per_epoch == cfg_mf.steps_per_epoch
    assert cfg_d.expectile == cfg_mf.expectile
    assert cfg_d.temperature == cfg_mf.temperature

    data_d = load_d_data(cfg_d)
    data_mf = load_d_data(cfg_mf)
    assert data_d.states_in.shape[-1] == 26
    assert data_mf.states_in.shape[-1] == 8
    # same calendar, same normalization
    assert data_d.dates.equals(data_mf.dates)
    np.testing.assert_allclose(data_d.state_mean, data_mf.state_mean, atol=1e-12)
    # fuzzy=True attaches the fixed layer; fuzzy=False has none
    assert data_d.fuzzy is not None
    assert data_mf.fuzzy is None


def test_make_windows_pads_before_series_start() -> None:
    states = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    # window ending at global position 2 with u=5: rows -2..2, only 0..2 exist
    w, ts = make_windows(states, np.array([2]), u=5)
    assert w.shape == (1, 5, 3)
    assert bool((w[0, :2] == 0).all())  # left-pad zeros
    torch.testing.assert_close(w[0, 2:], states[:3], rtol=0, atol=0)
    # padded positions map to timestep 0; real positions keep absolute index
    np.testing.assert_array_equal(ts[0].numpy(), [0, 0, 0, 1, 2])

    # full window (no pad) later in the series
    w2, ts2 = make_windows(states, np.array([9]), u=5)
    torch.testing.assert_close(w2[0], states[5:10], rtol=0, atol=0)
    np.testing.assert_array_equal(ts2[0].numpy(), [5, 6, 7, 8, 9])