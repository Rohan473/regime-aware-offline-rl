"""Conceptual figure: information -> exposure pathway (hybrid result).

Solid boxes = established findings; the exposure -> hybrid gate -> utility chain
is the mechanism supported by the hybrid experiment. Output:
paper/hybrid_mechanism.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, text, w=2.6, h=0.9, fc="#eef3fb", ec="#3b5b8c"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5)


def arrow(ax, p, q, dashed=False, color="#333333"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12,
                                 lw=1.2, color=color,
                                 linestyle="--" if dashed else "-",
                                 shrinkA=2, shrinkB=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box(ax, 5, 9.3, "32 point-in-time features", fc="#dfe9f7")
    box(ax, 5, 8.1, "Learned representation h_t")
    box(ax, 2.3, 6.6, "Directional information\n~ chance", fc="#f6e3e3", ec="#8c3b3b")
    box(ax, 7.7, 6.6, "Risk-state information\nOOD / vol / drawdown", fc="#e6f3e6", ec="#3b8c4f")
    box(ax, 2.3, 5.2, "No measurable\ndirectional skill", fc="#f6e3e3", ec="#8c3b3b")
    box(ax, 7.7, 5.2, "Adaptive exposure e_t", fc="#e6f3e6", ec="#3b8c4f")
    box(ax, 7.7, 3.9, "Risk-aware gate alpha_t", fc="#e6f3e6", ec="#3b8c4f")
    box(ax, 5, 2.7, "a_t = e_t + alpha_t(1-e_t)", w=3.4)
    box(ax, 5, 1.2, "Improved risk-adjusted utility", w=3.4, fc="#eef7ee", ec="#3b8c4f")

    arrow(ax, (5, 8.85), (5, 8.55))
    for x in (2.3, 7.7):
        arrow(ax, (5, 7.65), (x, 7.05))
    arrow(ax, (2.3, 6.15), (2.3, 5.65))     # directional -> terminal (no arrow to hybrid)
    arrow(ax, (7.7, 6.15), (7.7, 5.65))     # risk -> adaptive exposure
    arrow(ax, (7.7, 4.75), (7.7, 4.35))     # adaptive -> gate
    arrow(ax, (7.0, 3.55), (5.8, 3.0))      # gate -> a_t
    arrow(ax, (5, 2.25), (5, 1.65))         # a_t -> utility

    ax.text(0.2, 0.35, "Directional branch TERMINATES (no measurable skill); the hybrid uses only risk-state information.",
            fontsize=7.5, color="#555555")
    fig.tight_layout()
    out = ROOT / "paper" / "hybrid_mechanism.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()