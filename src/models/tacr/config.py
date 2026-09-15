"""Model C (TACR) configuration.

Reproduces Lee & Moon, "Transformer Actor-Critic with Regularization:
Automated Stock Trading using Reinforcement Learning" (IEEE Access 2023,
DOI 10.1109/ACCESS.2023.3324458; authors' code github.com/VarML/TACR) on
the project's continuous-action market.

Values mirror the authors' single-index (NDX/MDAX/CSI) setup unless the
project spec overrides them (see ``configs/tacr.yaml`` and the deviations
list in the package docstring):
  - u=20 (the paper's context length for single-index daily datasets) ==
    ``WINDOW_DAYS`` so Model C sees the same information horizon as B.
  - L decoder blocks default 4 (user-specified range 3-4; the paper used 5).
  - alpha=0.9 BC-regularization weight (paper's NDX/MDAX/CSI value).
  - critic learning rate 1e-6, gamma 0.99, tau 0.005, AdamW 1e-4 + wd 1e-4,
    gradient clip 0.25, batch 64 (all from the paper). Warmup and total step
    budget are scaled for CPU (10k -> 1k warmup; 40k -> 3k steps); see
    ``configs/tacr.yaml`` and the deviations list in the package docstring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.models import WINDOW_DAYS

CONFIG_YAML = Path(__file__).resolve().parents[3] / "configs" / "tacr.yaml"


@dataclass
class TACRConfig:
    # --- architecture (paper values; L per user spec) ---
    u: int = WINDOW_DAYS                  # context length in MDP triples
    embed_dim: int = 128                  # paper
    n_layer: int = 4                      # paper 5; user spec default 3-4 (deviation)
    n_head: int = 1                       # paper
    n_inner: int = 512                    # paper: 4 * embed_dim
    dropout: float = 0.1                  # paper
    max_ep_len: int = 5283                # longest trajectory (timestep embedding)
    action_head: str = "tanh"             # paper: linear+softmax (discrete); ours: linear+tanh (continuous)

    # --- training (paper values; budget/warmup scaled for CPU, see yaml) ---
    epochs: int = 10                      # val-eval cadence (3k steps total; paper 40k)
    steps_per_epoch: int = 300
    batch_size: int = 64                  # paper
    lr: float = 1e-4                      # paper
    weight_decay: float = 1e-4            # paper
    warmup_steps: int = 1000              # paper 10k (40k-step schedule); scaled to 3k steps
    grad_clip: float = 0.25               # paper
    alpha: float = 0.9                    # paper's BC-regularization weight (NDX/MDAX/CSI)
    gamma: float = 0.99                   # paper
    tau: float = 0.005                    # paper
    critic_lr: float = 1e-6               # paper default for NDX/MDAX/CSI
    seed: int = 20260814

    # --- structural fixes (defaults OFF = paper-faithful objective). Each
    #     targets the documented Q-collapse mechanism (PROJECT_NOTES 7.6),
    #     NOT hyperparameters: A = clipped double Q, B = BCQ-style hard
    #     action constraint, C = PAR direction-aware BC target replacement.
    use_double_q: bool = False     # A: min of two critic targets (TD3-style)
    double_q_mode: str = "min"     #    double-Q aggregation: "min" (TD3) | "mean" (ensemble)
    use_bcq: bool = False          # B: hard constraint around a generative action model
    bcq_phi: float = 0.5           #    max |action - generated| (data-support radius)
    bcq_bc_coeff: float = 0.0      #    BC weight alongside the constraint (0 = constraint only)
    use_par: bool = False          # C: direction-aware BC target replacement
    par_phi: float = 0.5           #    projection radius around the logged action
    par_cos_thresh: float = 0.0    #    replace BC target when cos(Q-grad, BC-dir) < thresh
    par_bc_coeff: float = 1.0      #    BC weight applied to the (possibly replaced) target
    action_model_hidden: int = 64  #    generative model width (fix B)
    action_model_steps: int = 2000 #    pretrain steps for the generative model
    action_model_lr: float = 1e-3
    action_model_batch: int = 256

    # --- behavior-policy selection ---
    # Which logged policy trajectories TACR trains on. Default () = ALL
    # policies in the offline dataset (the window-expanded families +
    # buy_and_hold + random). Set e.g. ("random",) to exclude the uniform
    # random demonstrator from the BC anchor / critic support.
    exclude_policies: tuple[str, ...] = ()
    # Family-balanced batch sampler: FIRST pick a family uniformly (1/N),
    # THEN a policy/window inside it, so every family contributes equally
    # to the BC prior regardless of window count (offsets the mean_reversion
    # dominance under uniform policy sampling). Default False = paper's
    # uniform-over-policies sampler.
    balanced_families: bool = False

    # --- eval / artifacts ---
    rtg_target: float = 0.0               # constant return-to-go at roll time (paper feeds zeros)
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).parent / "checkpoints")
    device: str = "cpu"

    # --- state widening (Exp 1, PROJECT_NOTES 7.30.1; default OFF) ---
    # When True, the TACR STATE becomes 16-dim = [8 SPY z] + [8 macro causal z]
    # (PROJECT_NOTES 7.18/7.30.1). Dates are restricted AT LOAD TIME to rows
    # where every macro feature is finite (2007-05+, HYG inception), mirroring
    # the C+ sign model's matched dates. Val/test fall entirely inside 2007+,
    # so only the TRAIN window loses its pre-2007 prefix. All other
    # hyperparameters are IDENTICAL to the canonical 8-dim run. Checkpoints
    # write to checkpoints/tacr/macro16/s{seed}/ so the canonical pack is
    # untouched (the pre-registered Exp 1 constraint).
    state_macro: bool = False

    @classmethod
    def from_yaml(cls, path: Path | str = CONFIG_YAML) -> "TACRConfig":
        """Build from configs/tacr.yaml (unknown keys ignored)."""
        from omegaconf import OmegaConf

        raw = OmegaConf.load(str(path))
        d = asdict(cls())
        d.update({k: v for k, v in raw.model.items() if k in d})
        d.update({k: v for k, v in raw.training.items() if k in d})
        d.update({k: v for k, v in raw.eval.items() if k in d})
        if not isinstance(d["exclude_policies"], tuple):
            d["exclude_policies"] = tuple(d["exclude_policies"])
        cfg = cls(**d)
        cfg.checkpoint_dir = Path(cfg.checkpoint_dir)
        if not cfg.checkpoint_dir.is_absolute():
            cfg.checkpoint_dir = Path(__file__).parent / cfg.checkpoint_dir
        cfg.validate()
        return cfg

    def validate(self) -> None:
        problems = []
        if self.u < 2:
            problems.append("u (context length) must be >= 2")
        if self.u != WINDOW_DAYS:
            problems.append(
                f"u={self.u} deviates from WINDOW_DAYS={WINDOW_DAYS} — the B/C "
                "comparison requires identical information horizons"
            )
        if self.embed_dim < 8:
            problems.append("embed_dim must be >= 8")
        if self.n_layer < 1:
            problems.append("n_layer must be >= 1")
        if self.n_head < 1:
            problems.append("n_head must be >= 1")
        if self.action_head not in ("tanh", "gaussian"):
            problems.append("action_head must be 'tanh' or 'gaussian'")
        if self.double_q_mode not in ("min", "mean"):
            problems.append("double_q_mode must be 'min' or 'mean'")
        if self.epochs < 1 or self.steps_per_epoch < 1:
            problems.append("epochs and steps_per_epoch must be >= 1")
        if self.batch_size < 1:
            problems.append("batch_size must be >= 1")
        if self.lr <= 0 or self.critic_lr <= 0:
            problems.append("lr and critic_lr must be > 0")
        if not 0.0 <= self.alpha <= 10.0:
            problems.append("alpha must be in [0, 10]")
        if not 0.0 < self.gamma < 1.0:
            problems.append("gamma must be in (0, 1)")
        if not 0.0 < self.tau <= 1.0:
            problems.append("tau must be in (0, 1]")
        if self.max_ep_len < self.u:
            problems.append("max_ep_len must be >= u")
        if self.bcq_phi <= 0 or self.par_phi <= 0:
            problems.append("bcq_phi and par_phi must be > 0")
        if not -1.0 <= self.par_cos_thresh <= 1.0:
            problems.append("par_cos_thresh must be in [-1, 1]")
        if self.bcq_bc_coeff < 0 or self.par_bc_coeff < 0:
            problems.append("bcq_bc_coeff and par_bc_coeff must be >= 0")
        if self.action_model_steps < 1 or self.action_model_batch < 1:
            problems.append("action_model_steps and action_model_batch must be >= 1")
        if problems:
            raise ValueError("invalid TACRConfig: " + "; ".join(problems))
