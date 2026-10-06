"""Styled Figure 1: the decision-relevant risk/exposure pathway.

Clean boxed/colored diagram (no monospace): representation -> directional
information (terminates, ~chance) and risk-state information -> adaptive
exposure -> risk-aware hybrid -> risk-adjusted utility. Output:
paper/fig1_pathway.png.
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

BLUE = ("#e8eefb", "#2f4e7e")
GREEN = ("#e5f4e8", "#2f7d46")
RED = ("#fbe9e9", "#8f3b3b")
GREY = ("#f2f2f2", "#666666")


def box(ax, x, y, text, fc, ec, w=2.7, h=0.95, fs=9, bold_first=True, lw=1.3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.03,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3,
            color="#111111", linespacing=1.35)


def arrow(ax, p, q, color="#333333", dashed=False, lw=1.5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14,
                                 lw=lw, color=color,
                                 linestyle="--" if dashed else "-",
                                 shrinkA=3, shrinkB=3, zorder=1))


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.4)
    ax.axis("off")

    box(ax, 5, 7.7, "32 point-in-time features", *BLUE, w=3.4, h=0.7, fs=9.5)
    box(ax, 5, 6.6, "Learned representation  h$_t$", *BLUE, w=4.0, h=0.8, fs=10)

    box(ax, 2.4, 5.3, "Directional information\n~ chance", *RED, w=3.0, h=1.0)
    box(ax, 7.6, 5.3, "Risk-state information\nOOD / volatility / drawdown",
        *GREEN, w=3.4, h=1.0)

    box(ax, 2.4, 3.7, "No measurable\ndirectional skill", *RED, w=2.9, h=0.9,
        lw=1.6)

    box(ax, 7.6, 3.9, "Adaptive exposure  e$_t$", *GREEN, w=3.0, h=0.85)
    box(ax, 7.6, 2.6, "Risk-aware hybrid\n$a_t = e_t + \\alpha_t(1 - e_t)$",
        *GREEN, w=3.4, h=1.0)
    box(ax, 5, 1.2, "Improved risk-adjusted utility", *GREEN, w=4.0, h=0.85, fs=9.5)

    ax.text(5, 5.3, "and", ha="center", va="center", fontsize=8, color="#888888")

    arrow(ax, (5, 7.35), (5, 7.0))
    arrow(ax, (5, 6.2), (2.4, 5.8), color=RED[1])
    arrow(ax, (5, 6.2), (7.6, 5.8), color=GREEN[1])
    arrow(ax, (2.4, 4.8), (2.4, 4.15), color=RED[1], dashed=True)   # terminates
    arrow(ax, (7.6, 4.8), (7.6, 4.33), color=GREEN[1])
    arrow(ax, (7.6, 3.48), (7.6, 3.10), color=GREEN[1])
    arrow(ax, (6.4, 2.2), (5.9, 1.62), color=GREEN[1])

    ax.text(3.6, 4.55, "terminates", fontsize=7.5, color=RED[1], style="italic")
    ax.text(0.15, 0.15,
            "Decision algorithm remains the stronger robust determinant of utility (ANOVA, p < .0001); "
            "the hybrid advantage is consistent across tested markets\nand seed-robust on SPY, "
            "but not statistically established when training and finite-test-path uncertainty are both considered.",
            fontsize=7, color="#555555", ha="left", va="bottom")

    fig.tight_layout()
    out = ROOT / "paper" / "fig1_pathway.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()