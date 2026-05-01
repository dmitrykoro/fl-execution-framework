"""
PID coefficient ablation plot: accuracy curves + cumulative FP per round.
"""
import csv as csv_mod
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

OUT_PATH = Path("out/pid_coeff_ablation.pdf")

CONFIGS = [
    ("P-only",          "out/04-10-2026_11-43-22", "#E53935", "-",              r"$K_i{=}0,\,K_d{=}0$"),
    ("PI",              "out/04-10-2026_11-46-17", "#FF9800", "--",             r"$K_i{=}0.5,\,K_d{=}0$"),
    ("PID ($K_i$=0.25)","out/04-10-2026_11-49-06", "#4CAF50", "-.",            r"$K_i{=}0.25,\,K_d{=}0.1$"),
    ("PID (default)",   "out/04-10-2026_11-51-58", "#2196F3", ":",             r"$K_i{=}0.5,\,K_d{=}0.1$"),
    ("PID ($K_i$=1.0)", "out/04-10-2026_11-54-47", "#9C27B0", (0,(4,1,1,1)), r"$K_i{=}1.0,\,K_d{=}0.1$"),
]


def load(path):
    rows = list(csv_mod.DictReader(open(path)))
    rounds     = [int(r["round"]) for r in rows]
    acc        = [float(r["average_accuracy_history"]) for r in rows]
    cumul_fp   = []
    running = 0
    for r in rows:
        running += int(r["fp_history"])
        cumul_fp.append(running)
    return rounds, acc, cumul_fp


fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(3.375, 2.1),
    gridspec_kw={"wspace": 0.42, "left": 0.11, "right": 0.97,
                 "top": 0.78, "bottom": 0.20}
)

legend_handles = []
for label, d, color, ls, coeff_str in CONFIGS:
    rounds, acc, cumul_fp = load(f"{d}/csv/round_metrics_0.csv")

    l, = ax1.plot(rounds, acc, color=color, lw=1.0, ls=ls, label=label)
    ax2.plot(rounds, cumul_fp, color=color, lw=1.0, ls=ls)
    legend_handles.append(l)

# ── ax1: accuracy ─────────────────────────────────────────────────────────────
ax1.set_xlim(1, 40)
ax1.set_ylim(60, 100)
ax1.yaxis.set_major_locator(mticker.MultipleLocator(10))
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax1.set_xlabel("Round", fontsize=6.5)
ax1.set_ylabel("Test Accuracy", fontsize=6.5)
ax1.set_title("(a) Accuracy", fontsize=7.5, pad=2)
ax1.tick_params(labelsize=5.5)
ax1.grid(True, alpha=0.25, linewidth=0.4)
ax1.axvline(7, color="#888", lw=0.7, ls="--", alpha=0.6)
ax1.text(7.4, 62, "removal\nstarts", fontsize=4.5, color="#666", va="bottom")

# annotate final accuracies
for label, d, color, ls, _ in CONFIGS:
    rows = list(csv_mod.DictReader(open(f"{d}/csv/round_metrics_0.csv")))
    final = float(rows[-1]["average_accuracy_history"])
    ax1.annotate(f"{final:.1f}%", xy=(40, final),
                 xytext=(3, 0), textcoords="offset points",
                 fontsize=4.5, color=color, va="center")

# ── ax2: cumulative FP ────────────────────────────────────────────────────────
ax2.set_xlim(1, 40)
ax2.set_ylim(0, None)
ax2.set_xlabel("Round", fontsize=6.5)
ax2.set_ylabel("Cumulative FP", fontsize=6.5)
ax2.set_title("(b) Benign Client Exclusion Errors", fontsize=7.5, pad=2)
ax2.tick_params(labelsize=5.5)
ax2.grid(True, alpha=0.25, linewidth=0.4)
ax2.axvline(7, color="#888", lw=0.7, ls="--", alpha=0.6)

# ── shared legend ─────────────────────────────────────────────────────────────
fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=3,
    fontsize=4.8,
    frameon=False,
    bbox_to_anchor=(0.52, 1.02),
    columnspacing=0.6,
    handlelength=1.8,
    handletextpad=0.25,
)

fig.savefig(OUT_PATH, bbox_inches="tight", dpi=300)
print(f"Saved: {OUT_PATH}")
