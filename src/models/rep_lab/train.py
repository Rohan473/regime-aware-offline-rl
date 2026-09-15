"""Training for the Representation Lab (Stage 1 + downstream Stage 2).

Stage 1 -- representation learners. Each objective trains the SAME encoder
with its own loss over the blocked decision-date grid (identical to the
Objective Lab trunk); best checkpoint per objective is selected on a
validation metric (negative MSE for auto/predictive, negative InfoNCE for
contrastive) and stored under checkpoints/reps/<obj>/s<seed>/best.pt.

  auto         MSE(h_t -> full flattened window)
  predictive   balanced MSE(h_t -> standardized future targets)
  contrastive  InfoNCE between two augmented views of the same window

Stage 2 -- downstream RL on FROZEN representations. One generic A2C head
(tanh-Gaussian policy + bootstrapped value, exactly the Objective Lab arm-C
form) is trained for a given representation feed; best by val Sharpe.
Input feed = frozen encoder h_t for learned reps, or the flat raw window.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.eval.regime_eval import sharpe_ratio
from src.models.ddr.data import DDRData, load_ddr_data, split_ddr_data
from src.models.objective_lab.train import _squashed_logp, _valid_mask
from src.models.rep_lab.config import PRED_TARGETS, RepLabConfig, tag_path
from src.models.rep_lab.model import DownstreamAgent, RawFeat, build_rep_objective
from src.models.rep_lab.targets import predictive_targets, split_targets

BEST = "best.pt"


def _block_iter(n: int, block: int):
    for lo in range(0, n, block):
        yield lo, min(lo + block, n)


# --------------------------------------------------------------------------
# Stage 1 - representation learners
# --------------------------------------------------------------------------

def _augment_view(x: torch.Tensor, cfg: RepLabConfig, rng: torch.Generator,
                  fea_mask: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
    xv = x.clone()
    xv[fea_mask] = 0.0
    xv[time_mask] = 0.0
    xv = xv + torch.randn_like(xv, generator=rng) * cfg.noise_std
    return xv


def _make_masks(x: torch.Tensor, cfg: RepLabConfig, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    B, _, F = x.shape
    fea = torch.rand(B, cfg.window, F, generator=rng) < cfg.fea_mask_prob
    time_mask = torch.rand(B, cfg.window, generator=rng) < cfg.time_mask_prob
    return fea, time_mask


def train_representation(objective: str, cfg: RepLabConfig,
                         data: DDRData | None = None) -> tuple:
    from .config import OBJECTIVES
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if data is None:
        data = load_ddr_data(cfg.window)
    splits = split_ddr_data(data)
    train, val = splits["train"], splits["val"]
    tmask = _valid_mask(train)
    tw, tr = train.windows[tmask], train.next_returns[tmask]

    rep = build_rep_objective(objective, cfg)
    opt = torch.optim.Adam(rep.parameters(), lr=cfg.lr)
    path = tag_path(cfg, objective)
    path.mkdir(parents=True, exist_ok=True)

    t_full, full_dates = predictive_targets(data)
    movie = {"t_full": t_full, "full_dates": full_dates}
    if objective == "predictive":
        ttr = split_targets(t_full, full_dates, train)[tmask]
        mu, sd = np.nanmean(ttr, axis=0), np.nanstd(ttr, axis=0) + 1e-8
        movie.update({"mean": mu, "std": sd})

    best_val, best_path = float("-inf"), None
    history = []
    for epoch in range(cfg.epochs):
        rep.train()
        rng = torch.Generator().manual_seed(cfg.seed + epoch)
        losses = []
        for lo, hi in _block_iter(len(tw), cfg.batch):
            x = tw[lo:hi]
            vmask = tmask[lo:hi]
            if objective == "auto":
                pred = rep.head(rep.encoder.encode(x))
                loss = torch.mean((pred - x.reshape(x.shape[0], -1)) ** 2)
            elif objective == "predictive":
                h = rep.encoder.encode(x)
                ttr = split_targets(t_full, full_dates, train)[tmask]  # (M, 4) masked-aligned
                w = np.isfinite(ttr)
                pred = rep.head(h)
                tgt = (ttr[lo:hi] - movie["mean"]) / movie["std"]
                tgt = torch.from_numpy(np.nan_to_num(tgt, nan=0.0))
                w = torch.from_numpy(w[lo:hi]).float()
                loss = torch.mean(((pred - tgt) ** 2) * w *
                                  torch.as_tensor(np.asarray(cfg.target_weights, dtype=np.float64)))
            else:  # contrastive
                fa, ta = _make_masks(x, cfg, rng)
                fb, tb = _make_masks(x, cfg, rng)
                za = rep.head(rep.encoder.encode(_augment_view(x, cfg, rng, fa, ta)))
                zb = rep.head(rep.encoder.encode(_augment_view(x, cfg, rng, fb, tb)))
                za = torch.nn.functional.normalize(za, dim=1)
                zb = torch.nn.functional.normalize(zb, dim=1)
                logits = (za @ zb.T) / cfg.tau
                loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(x)))
            if not torch.isfinite(loss).item():
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        rep.eval()
        torch.save({"objective": objective, "state_dict": rep.state_dict(),
                    "config": {k: (str(v) if isinstance(v, type) else v)
                               for k, v in cfg.__dict__.items()}}, path / "last.pt")
        v = _val_metric(objective, rep, val, cfg, movie)
        history.append({"epoch": epoch + 1, "objective": objective,
                        "train_loss": float(np.mean(losses)) if losses else float("nan"),
                        "val_metric": v})
        if np.isfinite(v) and v > best_val:
            best_val = v
            best_path = path / BEST
            torch.save({"objective": objective, "state_dict": rep.state_dict(),
                        "config": {k: (str(v) if isinstance(v, type) else v)
                                   for k, v in cfg.__dict__.items()},
                        "epoch": epoch + 1, "val_metric": float(v)}, best_path)
            print(f"[{objective}] epoch {epoch + 1}: val {v:.4f}  (new best)")
    return rep, pd_history(history, objective, path), best_path


def _val_metric(objective: str, rep: nn.Module, val: DDRData,
                cfg: RepLabConfig, movie: dict) -> float:
    mask = _valid_mask(val)
    if objective == "contrastive":
        # InfoNCE on the held-out split with the SAME augmentation seed
        rng = torch.Generator().manual_seed(cfg.seed)
        x = val.windows[mask]
        fa, ta = _make_masks(x, cfg, rng)
        fb, tb = _make_masks(x, cfg, rng)
        za = torch.nn.functional.normalize(rep.head(rep.encoder.encode(_augment_view(x, cfg, rng, fa, ta))), dim=1)
        zb = torch.nn.functional.normalize(rep.head(rep.encoder.encode(_augment_view(x, cfg, rng, fb, tb))), dim=1)
        logits = (za @ zb.T) / cfg.tau
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(x)))
        return -float(loss.item())
    with torch.no_grad():
        h = rep.encoder.encode(val.windows[mask])
        if objective == "auto":
            pred = rep.head(h)
            xf = val.windows[mask].reshape(val.windows[mask].shape[0], -1)
            return -float(torch.mean((pred - xf) ** 2).item())
        tv = split_targets(movie["t_full"], movie["full_dates"], val)[mask]
        y = (torch.from_numpy(tv) - torch.from_numpy(movie["mean"])) / torch.from_numpy(movie["std"])
        pred = rep.head(h)
        good = torch.isfinite(y)
        y = torch.nan_to_num(y, nan=0.0)
        return -float(torch.mean(((pred - y) ** 2)[good]).item())


def pd_history(history, objective, path):
    import pandas as pd
    df = pd.DataFrame(history)
    df.to_csv(path / "training_log.csv", index=False)
    return df


def load_rep(objective: str, cfg: RepLabConfig, checkpoint) -> nn.Module:
    rep = build_rep_objective(objective, cfg)
    rep.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
    rep.eval()
    return rep


def extract_rep(objective: str, cfg: RepLabConfig, checkpoint) -> object:
    from src.interpret.extract import Extracted
    data = load_ddr_data(cfg.window)
    rep = load_rep(objective, cfg, checkpoint)
    with torch.no_grad():
        H = rep.encoder.encode(data.windows).numpy()
    return Extracted(name=f"replab {objective} ({cfg.hidden}d)", dates=data.dates, H=H)


def raw_window_rep(cfg: RepLabConfig) -> object:
    """Raw-flattened window baseline (no learning)."""
    from src.interpret.extract import Extracted
    data = load_ddr_data(cfg.window)
    H = data.windows.numpy().reshape(len(data.dates), -1)
    return Extracted(name=f"raw window ({cfg.window * cfg.in_dim}d)", dates=data.dates, H=H)


# --------------------------------------------------------------------------
# Stage 2 - downstream RL on a frozen representation
# --------------------------------------------------------------------------

def _feed_fn(name: str, cfg: RepLabConfig, batch: torch.Tensor) -> torch.Tensor:
    if name == "raw":
        return batch.reshape(batch.shape[0], -1)
    raise KeyError(name)


def train_downstream(rep_name: str, cfg: RepLabConfig,
                     data: DDRData | None = None) -> tuple:
    """Train the generic A2C head on FROZEN features.

    rep_name == 'raw'   -> flat 20*8 raw window
    rep_name == any obj -> frozen encoder h_t for that objective
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if data is None:
        data = load_ddr_data(cfg.window)
    splits = split_ddr_data(data)
    train, val = splits["train"], splits["val"]
    tmask = _valid_mask(train)
    tw, tr = train.windows[tmask], train.next_returns[tmask]

    encoder = None
    if rep_name != "raw":
        ck = tag_path(cfg, rep_name) / BEST
        if not ck.exists():
            raise FileNotFoundError(f"{rep_name} representation not trained: {ck}")
        encoder = load_rep(rep_name, cfg, ck).encoder  # frozen

    feat = None if rep_name != "raw" else RawFeat(cfg.window * cfg.in_dim, cfg.hidden)
    agent = DownstreamAgent(cfg.hidden, feat)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr)
    path = tag_path(cfg, rep_name, subdir="downstream")
    path.mkdir(parents=True, exist_ok=True)

    def feed(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = encoder.encode(x) if encoder is not None else x.reshape(x.shape[0], -1)
        return z

    best_val, best_path = float("-inf"), None
    for epoch in range(cfg.epochs):
        agent.train()
        rng = torch.Generator().manual_seed(cfg.seed + epoch)
        losses = []
        for lo, hi in _block_iter(len(tw), 20):
            x, r = tw[lo:hi], tr[lo:hi]
            vmask = tmask[lo:hi]
            z = feed(x)
            z_next = torch.cat([z[1:], torch.zeros_like(z[:1])], dim=0)
            mu_raw = agent.mu_raw(z)
            mu = torch.tanh(mu_raw)
            with torch.no_grad():
                v_next = agent.v(z_next.detach())
            v_t = agent.v(z)
            r_t = r * mu
            pt = mu_raw + cfg.ac_sigma * torch.randn_like(mu_raw, generator=rng)
            a_s = torch.tanh(pt)
            logp = _squashed_logp(pt, mu_raw, cfg.ac_sigma, a_s)
            td = r_t + cfg.gamma * v_next.detach()
            A = (td - v_t).detach()
            pol = -torch.mean(logp[vmask] * A[vmask]) - cfg.ent_coef * torch.mean(logp[vmask])
            loss = pol + torch.mean((v_t[vmask] - td[vmask]) ** 2)
            if not torch.isfinite(loss).item():
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        agent.eval()
        v = downstream_sharpe(agent, feed, val, cfg)
        if np.isfinite(v) and v > best_val:
            best_val = v
            best_path = path / BEST
            torch.save({"rep": rep_name, "state_dict": agent.state_dict(),
                        "epoch": epoch + 1, "val_metric": float(v)}, best_path)
            print(f"[downstream {rep_name}] epoch {epoch + 1}: val Sharpe {v:.4f}  (new best)")
    if best_path is None:
        best_path = path / BEST
        torch.save({"rep": rep_name, "state_dict": agent.state_dict(),
                    "epoch": cfg.epochs, "val_metric": float("nan")}, best_path)
    return agent, None, best_path


def downstream_sharpe(agent: DownstreamAgent, feed, split: DDRData,
                      cfg: RepLabConfig) -> float:
    mask = _valid_mask(split)
    if int(mask.sum()) < 2:
        return float("nan")
    with torch.no_grad():
        z = feed(split.windows[mask])
        a = agent.mu(z).numpy()
    r = a * split.next_returns[mask].numpy()
    return sharpe_ratio(r, periods_per_year=252)


def supervised_direction_sharpe(X: np.ndarray, rsub, r_next: np.ndarray,
                                cost_bps: float = 0.0) -> float:
    """'Supervised' policy: sign of a logistic fit on train -> Sharpe on test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y = rsub["fwd_dir_1"].to_numpy()
    tr = (rsub["split"] == "train").to_numpy()
    te = (rsub["split"] == "test").to_numpy()
    good = np.isfinite(y) & np.isfinite(X).all(axis=1)
    Xtr, ytr = X[tr & good], y[tr & good]
    if len(ytr) < 10 or np.isclose((ytr > 0).mean(), 0) or np.isclose((ytr < 0).mean(), 0):
        return float("nan")
    Xte = X[te]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    yhat = clf.decision_function(sc.transform(Xte))
    r = np.sign(yhat) * r_next[te]
    r = r[np.isfinite(r)]
    return sharpe_ratio(r, periods_per_year=252)