"""Structural-fix variants (A/B/C) — unit tests for the helpers plus a
train-loop smoke test per fix.

A: clipped double Q — the pessimistic min of two critics is used for the
   TD target and the actor's Q (fixes Q overestimation collapse).
B: BCQ-style hard constraint — the constrained action stays within ``phi``
   of the generated action, and gradient flows only inside the band; the
   action model is trained, frozen, and saved in the checkpoint.
C: PAR — the BC target is replaced by a projection aligned with the
   Q-gradient and within ``phi`` of the logged action exactly when the two
   directions oppose; the projected target is detached (no second-order
   gradient).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.tacr.action_model import ConditionalGaussian, pretrain_action_model  # noqa: E402
from src.models.tacr.config import TACRConfig  # noqa: E402
from src.models.tacr.data import TACRData  # noqa: E402
from src.models.tacr.fixes import bcq_constrain, double_q, double_q_mean, double_q_min, par_bc_target  # noqa: E402
from src.models.tacr.train import train_tacr  # noqa: E402

torch.set_num_threads(1)


# ---------------------------------------------------------------- helpers ---

def _synthetic_data() -> TACRData:
    rng = np.random.default_rng(7)
    n = 1000
    dates = pd.DatetimeIndex(pd.bdate_range("2016-01-01", periods=n, freq="B"))
    states = rng.normal(size=(n, 4))
    actions = rng.uniform(-1.0, 1.0, size=(2, n))
    rewards = actions * 0.01
    rtgs = np.stack([np.cumsum(rewards[p][::-1])[::-1] for p in range(2)], axis=0)
    return TACRData(
        states=torch.tensor(states, dtype=torch.float32),
        actions=torch.tensor(actions, dtype=torch.float32),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        rtgs=torch.tensor(rtgs, dtype=torch.float32),
        timesteps=torch.arange(n, dtype=torch.long)[None, :].repeat(2, 1),
        dates=dates,
        regimes=pd.Series("bull", index=dates),
        valid=torch.ones(n, dtype=torch.bool),
        market_returns=torch.tensor(rng.normal(0.0, 0.01, size=n), dtype=torch.float32),
        policies=("p0", "p1"),
        state_mean=states.mean(axis=0),
        state_std=states.std(axis=0) + 1e-6,
    )


def _smoke_cfg(tmp_path: Path, **overrides) -> TACRConfig:
    kwargs = dict(
        u=20, embed_dim=32, n_layer=2, n_head=1, n_inner=128, dropout=0.0,
        max_ep_len=5283, epochs=2, steps_per_epoch=5, batch_size=8,
        warmup_steps=3, action_model_steps=50, seed=11,
        checkpoint_dir=tmp_path,
    )
    kwargs.update(overrides)
    return TACRConfig(**kwargs)


# -------------------------------------------------------------------- A ----

def test_double_q_min_uses_lower_of_two() -> None:
    q1 = torch.tensor([1.0, -0.5, 3.0])
    q2 = torch.tensor([0.8, 0.2, -1.0])
    assert torch.equal(double_q_min(q1, q2), torch.tensor([0.8, -0.5, -1.0]))


def test_double_q_mean_averages_and_dispatch() -> None:
    q1 = torch.tensor([1.0, -0.5, 3.0])
    q2 = torch.tensor([0.8, 0.2, -1.0])
    assert torch.allclose(double_q_mean(q1, q2), torch.tensor([0.9, -0.15, 1.0]))
    assert torch.equal(double_q(q1, q2, "min"), torch.minimum(q1, q2))
    assert torch.allclose(double_q(q1, q2, "mean"), (q1 + q2) / 2.0)


def test_double_q_mean_smoke(tmp_path) -> None:
    cfg = _smoke_cfg(tmp_path, use_double_q=True, double_q_mode="mean")
    model, hist, path = train_tacr(cfg, data=_synthetic_data())
    assert path.exists()
    assert np.isfinite(hist["actor_loss"]).all()


def test_double_q_critics_initialized_differently() -> None:
    from src.models.tacr.train import _build_critics

    torch.manual_seed(11)
    c1, c2 = _build_critics(4, 1, 2, seed=11)
    assert not np.allclose(
        c1.l1.weight.detach().numpy(), c2.l1.weight.detach().numpy()
    )  # must diverge or the min degenerates
    # first critic still reproducible at the same RNG state
    torch.manual_seed(11)
    c_single = _build_critics(4, 1, 1, seed=11)[0]
    assert np.allclose(c_single.l1.weight.detach().numpy(), c1.l1.weight.detach().numpy())


def test_double_q_smoke(tmp_path) -> None:
    cfg = _smoke_cfg(tmp_path, use_double_q=True)
    model, hist, path = train_tacr(cfg, data=_synthetic_data())
    assert path.exists()
    assert np.isfinite(hist["actor_loss"]).all()
    assert "q1_q2_spread" in hist.columns and not hist["q1_q2_spread"].isna().all()


# -------------------------------------------------------------------- B ----

def test_bcq_constrain_stays_within_phi() -> None:
    a_gen = torch.tensor([[0.2], [-0.3], [0.5]])
    pi = torch.tensor([[1.0], [0.9], [-1.0]])
    out = bcq_constrain(pi, a_gen, phi=0.5)
    assert torch.allclose(out, torch.tensor([[0.7], [0.2], [0.0]]))
    assert torch.all(torch.abs(out - a_gen) <= 0.5 + 1e-6)


def test_bcq_constrain_gradient_flows_inside_band_only() -> None:
    a_gen = torch.tensor([[0.0]])
    pi = torch.tensor([0.3], requires_grad=True)
    out = bcq_constrain(pi, a_gen, phi=0.5)
    out.sum().backward()
    assert pi.grad is not None and float(pi.grad.abs()) > 0.0  # unsaturated

    pi2 = torch.tensor([0.9], requires_grad=True)
    out2 = bcq_constrain(pi2, a_gen, phi=0.5)
    out2.sum().backward()
    assert pi2.grad is not None and float(pi2.grad.abs()) == 0.0  # saturated


def test_bcq_action_model_fit_and_freeze() -> None:
    data = _synthetic_data()
    split = data
    am = pretrain_action_model(
        ConditionalGaussian(4, hidden=16), split, steps=100, batch_size=32,
        lr=1e-2, rng=np.random.default_rng(1), seed=2,
    )
    # frozen + eval mode
    assert am.training is False
    assert all(not p.requires_grad for p in am.parameters())
    # generated actions stay within the logged range
    s = torch.tensor(data.states[:8], dtype=torch.float32)
    a_gen = am(s)
    assert float(a_gen.abs().max()) <= 1.5


def test_bcq_smoke_saves_action_model(tmp_path) -> None:
    cfg = _smoke_cfg(tmp_path, use_bcq=True)
    model, hist, path = train_tacr(cfg, data=_synthetic_data())
    assert path.exists()
    assert np.isfinite(hist["actor_loss"]).all()
    assert "bcq_mean_dev" in hist.columns and not hist["bcq_mean_dev"].isna().all()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "action_model" in payload  # must be saved for eval-time constraint


# -------------------------------------------------------------------- C ----

def test_par_replaces_target_on_opposing_directions() -> None:
    pi = torch.tensor([0.5, -0.2, 0.0])
    a_logged = torch.tensor([-0.5, 0.6, 0.1])
    g_q = torch.tensor([1.0, -1.0, 1.0])  # Q up / Q down / Q up
    target, cos, frac = par_bc_target(pi, a_logged, g_q, phi=0.5, cos_thresh=0.0)
    assert cos.tolist() == pytest.approx([-1.0, -1.0, 1.0])
    assert frac.item() == pytest.approx(2 / 3)
    # replaced: a_logged + phi * sign(g_q); aligned element unchanged
    assert torch.allclose(target[0], torch.tensor(0.0))   # -0.5 + 0.5*(+1)
    assert torch.allclose(target[1], torch.tensor(0.1))   #  0.6 + 0.5*(-1)
    assert torch.allclose(target[2], torch.tensor(0.1))   # unchanged


def test_par_projection_within_phi_of_logged() -> None:
    rng = np.random.default_rng(0)
    pi = torch.tensor(rng.uniform(-1, 1, 200))
    a = torch.tensor(rng.uniform(-1, 1, 200))
    g = torch.tensor(rng.choice([-1.0, 1.0], size=200))
    target, _, _ = par_bc_target(pi, a, g, phi=0.5, cos_thresh=0.0)
    assert torch.all(torch.abs(target - a) <= 0.5 + 1e-6)


def test_par_smoke(tmp_path) -> None:
    cfg = _smoke_cfg(tmp_path, use_par=True)
    model, hist, path = train_tacr(cfg, data=_synthetic_data())
    assert path.exists()
    assert np.isfinite(hist["actor_loss"]).all()
    assert "par_frac_replaced" in hist.columns and not hist["par_frac_replaced"].isna().all()


def test_balanced_family_sampler_gives_each_family_one_n() -> None:
    """Family-balanced sampling: each family (momentum / mean_reversion /
    buy_and_hold) receives ~1/N of the draws regardless of window count,
    restoring the BC prior that uniform policy sampling destroys."""
    import collections
    import pandas as pd

    from src.models.tacr.data import _family_of, sample_batch

    n = 80
    policies = ("momentum_30d", "momentum_60d", "mean_reversion_5d",
                "mean_reversion_10d", "mean_reversion_20d", "buy_and_hold")
    dates = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=n))
    split = TACRData(
        states=torch.randn(n, 4),
        actions=torch.rand(6, n) * 2 - 1,
        rewards=torch.rand(6, n) * 0.01,
        rtgs=torch.rand(6, n),
        timesteps=torch.arange(n, dtype=torch.long)[None, :].repeat(6, 1),
        dates=dates,
        regimes=pd.Series("bull", index=dates),
        valid=torch.ones(n, dtype=torch.bool),
        market_returns=torch.randn(n),
        policies=policies,
        state_mean=np.zeros(4),
        state_std=np.ones(4),
    )
    assert [_family_of(p) for p in policies] == [
        "momentum", "momentum", "mean_reversion", "mean_reversion",
        "mean_reversion", "buy_and_hold",
    ]

    rng = np.random.default_rng(0)
    fam = collections.Counter()
    for _ in range(1000):
        b = sample_batch(split, u=6, batch_size=64, rng=rng, balanced_families=True)
        for pi in b["policy_idx"].tolist():
            fam[_family_of(policies[pi])] += 1
    total = float(sum(fam.values()))
    for f in ("momentum", "mean_reversion", "buy_and_hold"):
        assert abs(fam[f] / total - 1.0 / 3) < 0.05, f"{f}: {fam[f] / total}"

    # uniform (paper default) lets the 3-member mean_reversion family dominate
    fam_u = collections.Counter()
    for _ in range(1000):
        b = sample_batch(split, u=6, batch_size=64, rng=rng, balanced_families=False)
        for pi in b["policy_idx"].tolist():
            fam_u[_family_of(policies[pi])] += 1
    total_u = float(sum(fam_u.values()))
    assert fam_u["mean_reversion"] / total_u > 0.45  # 3 of 6 policies