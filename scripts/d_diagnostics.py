"""Model D post-hoc diagnostics — is the null ablation interpretable?

The 3k/20k ablation (D vs D-minus-fuzzy, delta << noise floor) is only
evidence AGAINST the fuzzy hypothesis if the trained policy actually uses
its input representation. A policy converged to a bland dataset-anchored
behavior would produce the identical null. Three checks:

1. INPUT-SENSITIVITY PERTURBATION TEST (the decisive one). On the test
   split, hold the roll protocol fixed and perturb ONLY parts of the
   input matrix, then measure how much the deterministic action series
   moves (corr vs baseline, mean|delta a|, Sharpe of the perturbed roll):
     - fuzzy-zero    : zero the 18 fuzzy channels (raw 8 kept)
     - fuzzy-shuffle : swap in another day's fuzzy values (raw 8 kept)
     - raw-shuffle   : swap in another day's raw 8 values (fuzzy kept)
     - all-shuffle   : swap in another day's entire state vector
   Timestep embeddings are unchanged (same window ends), so any action
   movement is attributable to state CONTENT. Readout:
     all-shuffle ~ 0  => input-insensitive policy => ablation UNINFORMATIVE
     all-shuffle >> fuzzy-shuffle ~ 0 => inputs used, fuzzy unused
     fuzzy-shuffle ~ all-shuffle => fuzzy channels dominate
   D-minus-fuzzy gets raw-shuffle (its only input).

2. BEHAVIOR COMPARISON + SEED-FLIP. Per-seed exposure (mean|a|, turnover,
   short_frac), action std, sign-agreement and cross-VARIANT action
   correlation (vs the within-variant cross-seed corr), and per-seed wins
   vs own-EM at 3k and 20k to identify which seed flipped for D.

3. COMPUTE-MATCH VERIFICATION. Equal batch_size / steps_per_epoch / epochs
   between D and D-minus-fuzzy at matched budgets, read from the configs
   SAVED IN EACH CHECKPOINT (not from the repo yaml).

Outputs: printed tables + src/models/d/checkpoints/diagnostics/*.csv
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.regime_eval import sharpe_ratio  # noqa: E402
from src.models.d.config import DConfig  # noqa: E402
from src.models.d.data import load_d_data, make_windows, split_d_data  # noqa: E402
from src.models.d.eval import SEEDS, VARIANT_DIRS  # noqa: E402
from src.models.d.train import BEST_NAME, load_agent  # noqa: E402

PERTURB_RNG_SEED = 123


def roll_states(agent, cfg, states, test) -> np.ndarray:
    valid_mask = test.valid.numpy()
    ends = test.global_pos[valid_mask]
    windows, ts = make_windows(states, ends, cfg.u)
    with torch.no_grad():
        return agent.policy(agent.h_last(windows, ts)).numpy()


def main() -> None:
    cfg0 = DConfig.from_yaml()
    out_dir = cfg0.checkpoint_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ================= 1. input-sensitivity perturbation test =================
    print("=" * 70)
    print("1. INPUT-SENSITIVITY PERTURBATION TEST (test split, best-val ckpts)")
    print("=" * 70)
    pert_rows = []
    for tag, budget in ((None, "3k"), ("d_20k", "20k")):
        base = cfg0.checkpoint_dir / (tag or "")
        for variant in ("D", "D-minus-fuzzy"):
            fuzzy = variant == "D"
            cfg = replace(cfg0, fuzzy=fuzzy)
            data = load_d_data(cfg)
            test = split_d_data(data)["test"]
            m = test.market_returns[test.valid.numpy()].numpy()
            states = data.states_in
            T = states.shape[0]
            rng = np.random.default_rng(PERTURB_RNG_SEED)
            perm = rng.permutation(T)

            variants_states = {"baseline": states}
            if fuzzy:
                fz = states.clone(); fz[:, 8:] = 0.0
                variants_states["fuzzy-zero"] = fz
                fs = states.clone(); fs[:, 8:] = states[perm, 8:]
                variants_states["fuzzy-shuffle"] = fs
                rs = states.clone(); rs[:, :8] = states[perm, :8]
                variants_states["raw-shuffle"] = rs
            else:
                rs = states.clone(); rs[:, :] = states[perm, :]
                variants_states["raw-shuffle"] = rs
            variants_states["all-shuffle"] = states[perm]

            for seed in SEEDS:
                cp = base / VARIANT_DIRS[variant] / f"s{seed}" / BEST_NAME
                agent = load_agent(cp, cfg)
                a_base = roll_states(agent, cfg, states, test)
                for name, st in variants_states.items():
                    a = a_base if name == "baseline" else roll_states(agent, cfg, st, test)
                    pert_rows.append({
                        "budget": budget, "variant": variant, "seed": seed,
                        "perturbation": name,
                        "corr_vs_base": round(float(np.corrcoef(a_base, a)[0, 1]), 4),
                        "mean_abs_delta_a": round(float(np.abs(a - a_base).mean()), 4),
                        "sharpe": round(float(sharpe_ratio(a * m)), 4),
                        "action_std": round(float(np.std(a)), 4),
                    })
    pert = pd.DataFrame(pert_rows)
    pert.to_csv(out_dir / "input_sensitivity.csv", index=False)
    agg = pert.groupby(["budget", "variant", "perturbation"]).agg(
        corr_mean=("corr_vs_base", "mean"), corr_min=("corr_vs_base", "min"),
        delta_mean=("mean_abs_delta_a", "mean"), sharpe_mean=("sharpe", "mean"),
    ).round(4)
    print(agg.to_string())
    print("readout: all-shuffle corr ~ 1 => policy ignores state content (ablation uninformative);")
    print("         fuzzy-shuffle ~ 1 while raw/all move => fuzzy channels unread (null is about fuzzy);")
    print("         fuzzy-shuffle ~ all-shuffle => fuzzy channels dominate the input.")

    # ================= 2. behavior comparison + seed flip =================
    print()
    print("=" * 70)
    print("2. BEHAVIOR COMPARISON + PER-SEED WINS vs EM (both budgets)")
    print("=" * 70)
    beh_rows, wins_rows = [], []
    flipped = []
    for tag, budget in ((None, "3k"), ("d_20k", "20k")):
        base = cfg0.checkpoint_dir / (tag or "")
        actions = {}
        for variant in ("D", "D-minus-fuzzy"):
            cfg = replace(cfg0, fuzzy=(variant == "D"))
            data = load_d_data(cfg)
            test = split_d_data(data)["test"]
            m = test.market_returns[test.valid.numpy()].numpy()
            for seed in SEEDS:
                cp = base / VARIANT_DIRS[variant] / f"s{seed}" / BEST_NAME
                agent = load_agent(cp, cfg)
                a = roll_states(agent, cfg, data.states_in, test)
                actions[(variant, seed)] = a
                beh_rows.append({
                    "budget": budget, "variant": variant, "seed": seed,
                    "mean_abs_a": round(float(np.abs(a).mean()), 4),
                    "short_frac": round(float((a < 0).mean()), 4),
                    "turnover": round(float(np.abs(np.diff(a)).mean()), 4),
                    "action_std": round(float(np.std(a)), 4),
                    "sharpe": round(float(sharpe_ratio(a * m)), 4),
                    "own_EM_sharpe": round(float(sharpe_ratio(np.abs(a) * m)), 4),
                })
                wins_rows.append({
                    "budget": budget, "variant": variant, "seed": seed,
                    "beats_EM": bool(sharpe_ratio(a * m) > sharpe_ratio(np.abs(a) * m)),
                })
        # cross-variant agreement per seed
        for seed in SEEDS:
            a_d, a_mf = actions[("D", seed)], actions[("D-minus-fuzzy", seed)]
            beh_rows.append({
                "budget": budget, "variant": "cross-variant", "seed": seed,
                "mean_abs_a": round(float(np.corrcoef(a_d, a_mf)[0, 1]), 4),
                "short_frac": round(float((np.sign(a_d) == np.sign(a_mf)).mean()), 4),
                "turnover": np.nan, "action_std": np.nan,
                "sharpe": np.nan, "own_EM_sharpe": np.nan,
            })
    beh = pd.DataFrame(beh_rows)
    beh.to_csv(out_dir / "behavior_comparison.csv", index=False)
    wins = pd.DataFrame(wins_rows).pivot_table(
        index=["variant", "seed"], columns="budget", values="beats_EM"
    )
    wins.to_csv(out_dir / "wins_vs_em.csv")
    print(wins.to_string())
    for variant in ("D", "D-minus-fuzzy"):
        w = wins.loc[variant]
        flip = [s for s in SEEDS if bool(w.loc[s, "3k"]) and not bool(w.loc[s, "20k"])]
        if flip:
            flipped.append((variant, flip))
    print(f"\nseed(s) that FLIPPED (won @3k, lost @20k): {flipped or 'none'}")
    print("\nexposure / behavior per seed (3k | 20k):")
    for variant in ("D", "D-minus-fuzzy"):
        sub = beh[(beh["variant"] == variant)][["budget", "seed", "mean_abs_a",
                                                  "short_frac", "turnover", "action_std", "sharpe"]]
        print(f"--- {variant} ---")
        print(sub.pivot_table(index="seed", columns="budget",
                              values=["mean_abs_a", "turnover", "action_std"]).round(4).to_string())
    print("--- cross-variant action corr (row: seed; mean_abs_a col = corr, short_frac = sign agreement) ---")
    cv = beh[beh["variant"] == "cross-variant"][["budget", "seed", "mean_abs_a", "short_frac"]]
    print(cv.pivot_table(index="seed", columns="budget",
                         values=["mean_abs_a", "short_frac"]).round(4).to_string())

    # ================= 2b. val landscape (direct flat/trendless check) ========
    print()
    print("=" * 70)
    print("2b. VAL LANDSCAPE, 20k runs (epochs 10-20): trendless plateau?")
    print("=" * 70)
    plat_rows = []
    for variant in ("D", "D-minus-fuzzy"):
        for seed in SEEDS:
            log = pd.read_csv(cfg0.checkpoint_dir / "d_20k" / VARIANT_DIRS[variant]
                              / f"s{seed}" / "training_log.csv")
            seg = log[(log.epoch >= 10) & (log.epoch <= 20)]
            best, mu, sd = log.val_sharpe.max(), seg.val_sharpe.mean(), seg.val_sharpe.std()
            plat_rows.append({
                "variant": variant, "seed": seed,
                "best_val": round(float(best), 3),
                "best_ep": int(log.val_sharpe.idxmax()) + 1,
                "plateau_mean_10_20": round(float(mu), 3),
                "plateau_std_10_20": round(float(sd), 3),
                "z_of_best": round(float((best - mu) / sd), 2),
            })
    plat = pd.DataFrame(plat_rows)
    plat.to_csv(out_dir / "val_landscape_20k.csv", index=False)
    print(plat.to_string(index=False))
    print("readout: plateau ~0.7 +- 0.13 for every seed; best-val peaks are 1.7-4.9")
    print("sigma noise spikes => best-val EPOCH is the largest noise draw, uninformative.")

    # ================= 3. compute-match verification =================
    print()
    print("=" * 70)
    print("3. COMPUTE-MATCH VERIFICATION (configs saved in each checkpoint)")
    print("=" * 70)
    cm_rows = []
    for tag, budget in ((None, "3k"), ("d_20k", "20k")):
        base = cfg0.checkpoint_dir / (tag or "")
        for variant in ("D", "D-minus-fuzzy"):
            for seed in SEEDS:
                cp = base / VARIANT_DIRS[variant] / f"s{seed}" / BEST_NAME
                pl = torch.load(cp, map_location="cpu", weights_only=False)
                c = pl["config"]
                cm_rows.append({
                    "budget": budget, "variant": variant, "seed": seed,
                    "epochs": c["epochs"], "steps_per_epoch": c["steps_per_epoch"],
                    "batch_size": c["batch_size"], "warmup": c["warmup_steps"],
                    "lr": c["lr"], "expectile": c["expectile"], "temperature": c["temperature"],
                    "fuzzy": c["fuzzy"], "state_in_dim": pl["state_in_dim"],
                })
    cm = pd.DataFrame(cm_rows)
    cm.to_csv(out_dir / "compute_match.csv", index=False)
    mismatch = cm.groupby("budget").nunique()
    ok = all(mismatch.loc[b, k] == 1 for b in ("3k", "20k")
             for k in ("epochs", "steps_per_epoch", "batch_size", "warmup", "lr",
                       "expectile", "temperature"))
    print(cm.groupby(["budget", "variant"]).first().to_string())
    print(f"\nbatch/steps/epochs/lr/IQL-params identical across variants at matched budgets: {ok}")
    print("(samples-per-step = batch_size transitions for BOTH variants; the (s, s') pair")
    print(" forward doubles encoder passes identically; only the input Linear width differs.)")

    print(f"\nall outputs under {out_dir}")


if __name__ == "__main__":
    main()