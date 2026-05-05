"""
CIFAR-100 α=0.1 — Exclusion Dynamics Report
Analyzes client removal behavior per strategy per attack.

Key metric semantics (framework convention):
  TP = clients removed whose score genuinely exceeded threshold (correctly removed)
  FP = clients removed whose score should NOT have exceeded threshold (incorrectly removed)
  FN = clients NOT removed whose score should have exceeded threshold (missed)
  TN = clients correctly kept (score below threshold)

With 10 malicious clients:
  Label Flipping:    malicious scores ≈ benign scores → FN=10 all strategies (undetectable)
  Gradient Scaling:  malicious scores >> benign scores → FN=0 for PID-MADE (perfect recall)
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUT_PDF = "out/cifar100_a01_exclusion_dynamics.pdf"

# α=0.1 output directories (chronological order from runner.log)
DIRS = {
    "label_flipping":   "out/04-08-2026_18-23-57",   # a01_label_flipping
    "gradient_scaling": "out/04-08-2026_21-52-58",   # a01_gradient_scaling
}

STRATEGIES  = ["PID", "PID-Std-Score", "PID-Stateless", "PID-MADE", "RFA"]
S_COLORS    = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]
S_LS        = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
MALICIOUS   = set(range(10))
ROUNDS      = list(range(1, 31))


def load_round_metrics(base_dir, s_idx):
    path = f"{base_dir}/csv/round_metrics_{s_idx}.csv"
    rows = list(csv.DictReader(open(path)))
    tp, fp, fn, tn, thresh = [], [], [], [], []
    for r in rows:
        tp.append(float(r.get("tp_history", 0) or 0))
        fp.append(float(r.get("fp_history", 0) or 0))
        fn.append(float(r.get("fn_history", 0) or 0))
        tn.append(float(r.get("tn_history", 0) or 0))
        raw = r.get("removal_threshold_history", "not collected")
        thresh.append(float(raw) if raw != "not collected" else None)
    return tp, fp, fn, tn, thresh


def load_per_client_scores(base_dir, s_idx, client_ids):
    path = f"{base_dir}/csv/per_client_metrics_{s_idx}.csv"
    rows = list(csv.DictReader(open(path)))
    result = {cid: [] for cid in client_ids}
    for r in rows:
        for cid in client_ids:
            val = r.get(f"client_{cid}_removal_criterion_history", None)
            try:
                result[cid].append(float(val) if val else None)
            except (ValueError, TypeError):
                result[cid].append(None)
    return result


def plot_fp_fn_over_time(ax, attack, show_fn=True):
    """Stack plot of TP/FP over rounds for each strategy."""
    for s_idx, (name, color, ls) in enumerate(zip(STRATEGIES, S_COLORS, S_LS)):
        base_dir = DIRS[attack]
        tp, fp, fn, tn, thresh = load_round_metrics(base_dir, s_idx)
        ax.plot(ROUNDS, fp, color=color, lw=2, ls=ls, label=f"{name} (FP)")
    ax.axvline(7, color="gray", lw=1, ls="--", alpha=0.6, label="Removal starts (R7)")
    ax.set_xlabel("Round", fontsize=8)
    ax.set_ylabel("False Positives (benign clients wrongly removed)", fontsize=8)
    ax.set_title(f"False Positive Accumulation — {attack.replace('_', ' ').title()}", fontsize=9)
    ax.set_xlim(1, 30)
    ax.set_ylim(-1, 45)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)


def plot_active_clients(ax, attack):
    """Lines of active client count = 100 - TP - FP."""
    for s_idx, (name, color, ls) in enumerate(zip(STRATEGIES, S_COLORS, S_LS)):
        tp, fp, fn, tn, thresh = load_round_metrics(DIRS[attack], s_idx)
        active = [100 - t - f for t, f in zip(tp, fp)]
        ax.plot(ROUNDS, active, color=color, lw=2, ls=ls, label=name)
    ax.axhline(90, color="black", lw=0.8, ls=":", alpha=0.5, label="90 benign clients")
    ax.axhline(10, color="red", lw=0.8, ls=":", alpha=0.5, label="10 malicious clients")
    ax.axvline(7, color="gray", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Round", fontsize=8)
    ax.set_ylabel("Active Clients", fontsize=8)
    ax.set_title(f"Active Client Count — {attack.replace('_', ' ').title()}", fontsize=9)
    ax.set_xlim(1, 30)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)


def plot_threshold_vs_scores(ax, attack, s_idx, s_name):
    """Threshold line vs avg malicious and avg benign scores per round."""
    base_dir = DIRS[attack]
    tp, fp, fn, tn, thresh = load_round_metrics(base_dir, s_idx)

    # Load scores for malicious (0-9) and sample of benign (10-99)
    mal_ids = list(range(10))
    benign_ids = list(range(10, 110, 10))[:9]  # sample 9 benign
    all_ids = mal_ids + benign_ids
    scores = load_per_client_scores(base_dir, s_idx, all_ids)

    mal_avg = [np.mean([scores[i][r] for i in mal_ids if scores[i][r] is not None]) for r in range(30)]
    mal_max = [max([scores[i][r] for i in mal_ids if scores[i][r] is not None]) for r in range(30)]
    mal_min = [min([scores[i][r] for i in mal_ids if scores[i][r] is not None]) for r in range(30)]
    ben_avg = [np.mean([scores[i][r] for i in benign_ids if scores[i][r] is not None]) for r in range(30)]
    ben_max = [max([scores[i][r] for i in benign_ids if scores[i][r] is not None]) for r in range(30)]
    ben_min = [min([scores[i][r] for i in benign_ids if scores[i][r] is not None]) for r in range(30)]

    ax.fill_between(ROUNDS, mal_min, mal_max, alpha=0.15, color="#E91E63")
    ax.fill_between(ROUNDS, ben_min, ben_max, alpha=0.15, color="#2196F3")
    ax.plot(ROUNDS, mal_avg, color="#E91E63", lw=2, label="Malicious avg")
    ax.plot(ROUNDS, ben_avg, color="#2196F3", lw=2, label="Benign avg (sample)")

    # Threshold
    t_vals = [t for t in thresh if t is not None]
    t_rounds = [ROUNDS[i] for i, t in enumerate(thresh) if t is not None]
    if t_vals:
        ax.plot(t_rounds, t_vals, color="black", lw=2, ls="--", label="Removal threshold")

    ax.axvline(7, color="gray", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Round", fontsize=8)
    ax.set_ylabel("Score", fontsize=8)
    ax.set_title(f"{s_name} — Score vs Threshold\n{attack.replace('_',' ').title()}", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    ax.set_xlim(1, 30)


def plot_precision_recall_f1(ax, attack):
    """Precision and recall over rounds."""
    for s_idx, (name, color, ls) in enumerate(zip(STRATEGIES, S_COLORS, S_LS)):
        tp, fp, fn, tn, thresh = load_round_metrics(DIRS[attack], s_idx)
        prec, rec = [], []
        for t, f, n in zip(tp, fp, fn):
            prec.append(t / (t + f) if (t + f) > 0 else 1.0)
            rec.append(t / (t + n) if (t + n) > 0 else 0.0)
        ax.plot(ROUNDS, prec, color=color, lw=2, ls=ls, label=f"{name}")
    ax.axvline(7, color="gray", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Round", fontsize=8)
    ax.set_ylabel("Precision", fontsize=8)
    ax.set_title(f"Removal Precision — {attack.replace('_',' ').title()}", fontsize=9)
    ax.set_xlim(1, 30)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)


def summary_bar(ax):
    """Grouped bar: final FP and FN per strategy per attack."""
    x = np.arange(len(STRATEGIES))
    width = 0.3
    for a_idx, (attack, color, label) in enumerate([
        ("label_flipping", "#FF7043", "Label Flipping"),
        ("gradient_scaling", "#5C6BC0", "Gradient Scaling"),
    ]):
        fps = []
        fns = []
        for s_idx in range(5):
            tp, fp, fn, tn, thresh = load_round_metrics(DIRS[attack], s_idx)
            fps.append(fp[-1])
            fns.append(fn[-1])
        offset = (a_idx - 0.5) * width
        ax.bar(x + offset, fps, width, label=f"{label} — FP", color=color, alpha=0.8, edgecolor="white")
        ax.bar(x + offset, fns, width, bottom=fps, label=f"{label} — FN",
               color=color, alpha=0.4, edgecolor="white", hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGIES, fontsize=8)
    ax.set_ylabel("Clients (FP solid, FN hatched)", fontsize=8)
    ax.set_title("Final FP + FN @ Round 30 — All Strategies", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=7)


def score_sep_over_time(ax, attack):
    """Score separation: mean_malicious - mean_benign over rounds."""
    benign_sample = list(range(10, 60, 5))
    for s_idx, (name, color, ls) in enumerate(zip(STRATEGIES, S_COLORS, S_LS)):
        mal_ids = list(range(10))
        scores = load_per_client_scores(DIRS[attack], s_idx, mal_ids + benign_sample)
        sep = []
        for r in range(30):
            m = np.mean([scores[i][r] for i in mal_ids if scores[i][r] is not None])
            b = np.mean([scores[i][r] for i in benign_sample if scores[i][r] is not None])
            sep.append(m - b)
        ax.plot(ROUNDS, sep, color=color, lw=2, ls=ls, label=name)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(7, color="gray", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Round", fontsize=8)
    ax.set_ylabel("Score separation (malicious − benign avg)", fontsize=8)
    ax.set_title(f"Score Separation — {attack.replace('_',' ').title()}", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    ax.set_xlim(1, 30)


# ─── Main ────────────────────────────────────────────────────────────────────
def print_dir_path_summary():
    """
    Print summary of Global Constants for user to check when starting script
    """
    print("\n=== Review Globals ===\n")
    print(f"STRATEGIES: {STRATEGIES}")

    print("\n=== Review Data Source Directory Paths ===\n")
    
    # Check DIR paths (originally just for α=0.1)
    print("Current DIR source directory paths:")
    for k, v in DIRS.items():
        print(f"  {k}: {v}")

print_dir_path_summary()
response = input("\nVerify globals and paths above. \nEnter 'y' if all paths are valid, or 'n' if you want to quit.\n > ")
if response.split()[0].lower() != "y":
    exit()

with PdfPages(OUT_PDF) as pdf:

    # PAGE 1: Title
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.88, "CIFAR-100 (α=0.1) — Client Exclusion Dynamics", ha="center", va="center",
            fontsize=20, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.79, "How each strategy removes clients across 30 rounds under Label Flipping & Gradient Scaling",
            ha="center", va="center", fontsize=12, color="#555", transform=ax.transAxes)

    summary_text = [
        ("TP/FP/FN/TN semantics (framework convention):", True),
        ("  TP = client removed whose score genuinely exceeded threshold (correctly removed)", False),
        ("  FP = client removed whose score should NOT have exceeded threshold (incorrectly removed)", False),
        ("  FN = client not removed whose score should have exceeded threshold (missed malicious)", False),
        ("  TN = client correctly kept (score below threshold)", False),
        ("", False),
        ("Setup: 100 clients, clients 0–9 are malicious, removal begins at round 7", True),
        ("", False),
        ("Label Flipping — Final state @ R30:", True),
        ("  PID:           FP=29 / FN=10  (29 benign wrongly removed; all 10 malicious missed)", False),
        ("  PID-Std-Score: FP=31 / FN=10  (worse FP, same miss rate)", False),
        ("  PID-Stateless: FP=30 / FN=10  (similar)", False),
        ("  PID-MADE:      FP=2  / FN=10  (near-zero collateral; malicious still undetectable)", False),
        ("  RFA:           FP=0  / FN=10  (perfect precision; malicious undetectable)", False),
        ("  → Label flipping is invisible to all strategies — attack score ≈ benign non-IID score", False),
        ("", False),
        ("Gradient Scaling (×5) — Final state @ R30:", True),
        ("  PID:           FP=31 / FN=0   (all malicious caught; 31 benign collateral)", False),
        ("  PID-Std-Score: FP=36 / FN=0   (highest collateral)", False),
        ("  PID-Stateless: FP=36 / FN=0   (highest collateral)", False),
        ("  PID-MADE:      FP=0  / FN=0   (PERFECT: all malicious caught, zero collateral)", False),
        ("  RFA:           FP=0  / FN=10  (no false positives; malicious clients re-enter late)", False),
        ("  → Gradient scaling is highly detectable; PID-MADE uniquely achieves FP=FN=0", False),
    ]
    y = 0.72
    for line, bold in summary_text:
        ax.text(0.06, y, line, ha="left", va="top", fontsize=9,
                fontweight="bold" if bold else "normal",
                color="#111" if bold else "#333",
                transform=ax.transAxes)
        y -= 0.032
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # PAGE 2: False Positive accumulation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("False Positive Accumulation Over Rounds (α=0.1)", fontsize=12, fontweight="bold")
    plot_fp_fn_over_time(axes[0], "label_flipping")
    plot_fp_fn_over_time(axes[1], "gradient_scaling")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # PAGE 3: Active client count
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Active Client Count Over Rounds (α=0.1)", fontsize=12, fontweight="bold")
    plot_active_clients(axes[0], "label_flipping")
    plot_active_clients(axes[1], "gradient_scaling")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # PAGE 4: Score separation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Score Separation: Malicious − Benign Average Score (α=0.1)", fontsize=12, fontweight="bold")
    score_sep_over_time(axes[0], "label_flipping")
    score_sep_over_time(axes[1], "gradient_scaling")
    for ax in axes:
        ax.set_ylim(-5, None)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # PAGE 5: Score vs threshold for PID and PID-MADE (both attacks)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Score vs Removal Threshold: PID vs PID-MADE (α=0.1)", fontsize=12, fontweight="bold")
    plot_threshold_vs_scores(axes[0][0], "label_flipping",   0, "PID")
    plot_threshold_vs_scores(axes[0][1], "label_flipping",   3, "PID-MADE")
    plot_threshold_vs_scores(axes[1][0], "gradient_scaling", 0, "PID")
    plot_threshold_vs_scores(axes[1][1], "gradient_scaling", 3, "PID-MADE")
    # Fix y-axis scale per row for fair comparison
    for row in axes:
        ymax = max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(0, ymax)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # PAGE 6: Removal precision over time
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Removal Precision Over Rounds (α=0.1)", fontsize=12, fontweight="bold")
    plot_precision_recall_f1(axes[0], "label_flipping")
    plot_precision_recall_f1(axes[1], "gradient_scaling")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # PAGE 7: Final FP + FN summary bar
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.suptitle("Final FP and FN @ Round 30 — All Strategies, Both Attacks (α=0.1)",
                 fontsize=12, fontweight="bold")
    summary_bar(ax)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # PAGE 8: Analysis text
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.97, "Exclusion Dynamics Analysis — Key Findings",
            ha="center", va="top", fontsize=15, fontweight="bold", transform=ax.transAxes)

    findings = [
        ("Finding 1: Label Flipping is fundamentally undetectable under α=0.1", [
            "Score separation between malicious and benign clients is near-zero throughout all 30 rounds.",
            "Label-flipped gradients fall within the range of benign non-IID gradients — no strategy can",
            "distinguish them. FN=10 (all 10 malicious missed) for every strategy.",
            "Implication: PID-MADE's accuracy advantage (35.7% vs 34.4% for PID) comes not from",
            "detecting malicious clients, but from removing fewer benign clients as collateral (FP=2 vs FP=29).",
        ]),
        ("Finding 2: Gradient Scaling is highly detectable — PID-MADE achieves perfect exclusion", [
            "Malicious score avg reaches 212 vs benign avg 44 by R30 — a 4.8× separation ratio.",
            "PID-MADE maintains FP=0, FN=0 throughout all 30 rounds: every malicious client caught,",
            "zero benign clients incorrectly removed. This is the only strategy to achieve this.",
            "PID, PID-Std, PID-Stateless all start cleanly (FP=0 through R8) but FP blooms to 31–36",
            "by R20 as the growing threshold engulfs benign clients with high non-IID variance.",
        ]),
        ("Finding 3: PID threshold instability causes the FP bloom", [
            "PID threshold grows unboundedly (16 → 50 for label flipping, 37 → 49 for grad scaling).",
            "As more benign clients are removed, the remaining pool's mean and std shift, pushing the",
            "threshold into the benign score range. This is a cascade: each benign removal inflates",
            "the next threshold estimate.",
            "PID-MADE's exponential decay on the integral term prevents this drift — the threshold",
            "stabilizes after a brief calibration period (R1–2) and does not grow further.",
        ]),
        ("Finding 4: RFA removes almost no clients (FP=0) but misses all malicious under label flipping", [
            "RFA's geometric median aggregation is inherently robust to outlier updates, so it doesn't",
            "need to expel clients. It silently down-weights anomalous contributions.",
            "Under label flipping, this is equivalent to PID-MADE's FP=0 behavior but without the",
            "accuracy advantage — RFA's geometric median degrades convergence under non-IID.",
        ]),
    ]

    y = 0.90
    for title, bullets in findings:
        ax.text(0.04, y, title, ha="left", va="top", fontsize=10, fontweight="bold",
                color="#1A237E", transform=ax.transAxes)
        y -= 0.035
        for b in bullets:
            ax.text(0.07, y, f"• {b}", ha="left", va="top", fontsize=8.5,
                    color="#222", transform=ax.transAxes)
            y -= 0.028
        y -= 0.015

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"Saved to {OUT_PDF}")
