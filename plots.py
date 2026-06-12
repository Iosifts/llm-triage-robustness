#!/usr/bin/env python3
"""
Regenerate all paper figures from the scored CSVs (reproducible end-to-end).

Usage:
  python plots.py

Reads:
  - summary_flips.csv
  - summary_consistency.csv

Writes PNG and PDF figures to:
  - figures/fig1_flip_by_condition
  - figures/fig2_consistency_vs_flip
  - figures/fig3_harm_direction
  - figures/fig4_heatmap
  - figures/fig5_radar_flip_by_model
  - figures/fig6_radar_harm_by_condition

Important IEEE sizing:
  - One-column figures are generated close to 3.45 inches wide.
  - Two-column figures are generated close to 7.16 inches wide.
  - This avoids tiny fonts after LaTeX scales the figure.
"""

import csv
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# ---------------------------------------------------------------------
# Paths and style
# ---------------------------------------------------------------------

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# IEEE approximate figure widths in inches.
IEEE_COL_WIDTH = 3.45
IEEE_TEXT_WIDTH = 7.16

# Colourblind-safe palette, close to Okabe-Ito.
MODEL_COLORS = [
    "#0072B2",  # blue
    "#009E73",  # green
    "#E69F00",  # orange
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue, fallback
    "#F0E442",  # yellow, fallback
]

SELF_COLOR = "#0072B2"
PERTURB_COLOR = "#D55E00"

DOWNGRADE_COLOR = "#D55E00"
OVERESC_COLOR = "#009E73"

HEATMAP_CMAP = "Oranges"

# These font sizes are intentionally modest because figures are now generated
# close to their final IEEE size. Do not use slide-sized figures and then shrink
# them in LaTeX, because the fonts will become unreadable.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "--",
    "figure.dpi": 120,
    "savefig.dpi": 300,
})


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run: python run_and_score.py --metrics-only"
        )
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def short_model_name(model):
    """Shorten Hugging Face model names for figures."""
    name = model.split("/")[-1]
    replacements = {
        "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
        "DeepSeek-R1-Distill-Qwen-7B": "DeepSeek-R1-7B",
        "gemma-2-9b-it": "Gemma-2-9B",
        "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
        "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    }
    return replacements.get(name, name)


def pretty_condition(condition):
    mapping = {
        "reorder": "reorder",
        "add_lowstakes": "low-stakes",
        "add_highstakes": "high-stakes",
        "compress": "compress",
        "expand": "expand",
        "paraphrase": "paraphrase",
        "original": "original",
    }
    return mapping.get(condition, condition.replace("_", "-"))


def savefig(name):
    """Save every figure as PNG and PDF with minimal margins."""
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    plt.tight_layout(pad=0.4)
    plt.savefig(png, bbox_inches="tight", pad_inches=0.03)
    plt.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"Wrote {png} and {pdf}")


def get_rate(rows, model, condition, key):
    for r in rows:
        if r["model"] == model and r["condition"] == condition:
            return float(r[key])
    return 0.0


def get_matrix(rows, models, conditions, key):
    return np.array([
        [get_rate(rows, model, condition, key) for condition in conditions]
        for model in models
    ])


def radar_angles(n):
    """Return closed radar angles."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.concatenate([angles, angles[:1]])


def close_values(values):
    values = np.asarray(values, dtype=float)
    return np.concatenate([values, values[:1]])


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

flips = read_csv("summary_flips.csv")
cons = read_csv("summary_consistency.csv")

models = sorted({r["model"] for r in flips})

conditions = []
for r in flips:
    c = r["condition"]
    if c not in conditions:
        conditions.append(c)

# Keep original out of perturbation plots if it appears.
conditions = [c for c in conditions if c != "original"]

model_labels = [short_model_name(m) for m in models]
condition_labels = [pretty_condition(c) for c in conditions]

M_flip = get_matrix(flips, models, conditions, "flip_rate")
M_down = get_matrix(flips, models, conditions, "safety_downgrade_rate")
M_up = get_matrix(flips, models, conditions, "over_escalation_rate")


# ---------------------------------------------------------------------
# Fig 1 - flip rate by condition, grouped by model
# Use as backup / appendix. The heatmap is better for the main paper.
# ---------------------------------------------------------------------

x = np.arange(len(conditions))
bar_width = 0.8 / max(len(models), 1)

plt.figure(figsize=(IEEE_TEXT_WIDTH, 3.15))
for i, model in enumerate(models):
    plt.bar(
        x + i * bar_width,
        M_flip[i],
        bar_width,
        label=short_model_name(model),
        color=MODEL_COLORS[i % len(MODEL_COLORS)],
    )

plt.xticks(
    x + bar_width * (len(models) - 1) / 2,
    condition_labels,
    rotation=18,
    ha="right",
)
plt.ylabel("verdict flip rate")
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
plt.title("Flip rate under meaning-preserving perturbations")
plt.legend(frameon=True, ncol=1)
savefig("fig1_flip_by_condition")


# ---------------------------------------------------------------------
# Fig 2 - self-inconsistency vs mean perturbation flip
# Use as backup / appendix unless you explicitly discuss stochasticity.
# ---------------------------------------------------------------------

mean_flip = {m: float(np.mean(M_flip[i])) for i, m in enumerate(models)}
self_inconsistency = {
    r["model"]: float(r["self_inconsistency_rate"])
    for r in cons
}

xm = np.arange(len(models))

plt.figure(figsize=(IEEE_TEXT_WIDTH, 3.15))
plt.bar(
    xm - 0.2,
    [self_inconsistency.get(m, 0.0) for m in models],
    0.4,
    label="self-inconsistency (identical input)",
    color=SELF_COLOR,
)
plt.bar(
    xm + 0.2,
    [mean_flip[m] for m in models],
    0.4,
    label="mean flip (perturbed input)",
    color=PERTURB_COLOR,
)

plt.xticks(xm, model_labels, rotation=14, ha="right")
plt.ylabel("rate")
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
plt.title("Instability floor vs perturbation-induced instability")
plt.legend(frameon=True)
savefig("fig2_consistency_vs_flip")


# ---------------------------------------------------------------------
# Fig 3 - direction of harm per condition, averaged over models
# Recommended as one-column main-paper figure.
# ---------------------------------------------------------------------

avg_down = M_down.mean(axis=0)
avg_up = M_up.mean(axis=0)
xc = np.arange(len(conditions))

plt.figure(figsize=(IEEE_COL_WIDTH, 2.55))
plt.bar(
    xc - 0.18,
    avg_down,
    0.36,
    label="safety downgrade",
    color=DOWNGRADE_COLOR,
)
plt.bar(
    xc + 0.18,
    avg_up,
    0.36,
    label="over-escalation",
    color=OVERESC_COLOR,
)

plt.xticks(xc, condition_labels, rotation=28, ha="right")
plt.ylabel("rate")
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
plt.title("Direction of instability")
plt.legend(frameon=True, loc="upper right", fontsize=7.5)
savefig("fig3_harm_direction")


# ---------------------------------------------------------------------
# Fig 4 - heatmap of flip rate, model x condition
# Recommended as two-column main-paper figure.
# In LaTeX, include this with figure* and width=\textwidth.
# ---------------------------------------------------------------------

plt.figure(figsize=(IEEE_TEXT_WIDTH, 3.15))
ax = plt.gca()

vmax = max(0.18, float(M_flip.max()) if M_flip.size else 0.18)
im = ax.imshow(M_flip, aspect="auto", cmap=HEATMAP_CMAP, vmin=0, vmax=vmax)

cbar = plt.colorbar(im)
cbar.set_label("flip rate")
cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
cbar.ax.tick_params(labelsize=8)

ax.set_xticks(range(len(conditions)))
ax.set_xticklabels(
    condition_labels,
    rotation=20,
    ha="right",
    fontsize=11,
)

ax.set_yticks(range(len(models)))
ax.set_yticklabels(
    model_labels,
    fontsize=11,
)

ax.set_xlabel("Perturbation", fontsize=11)
ax.set_ylabel("Model", fontsize=11)

for i in range(len(models)):
    for j in range(len(conditions)):
        value = M_flip[i, j]
        text_color = "white" if value > 0.115 else "black"
        ax.text(
            j,
            i,
            f"{100 * value:.1f}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=text_color,
        )

ax.set_title("Flip rate by model and perturbation")
savefig("fig4_heatmap")


# ---------------------------------------------------------------------
# Fig 5 - spider/radar plot: flip profile per model
# Use for slides or appendix, not main paper.
# ---------------------------------------------------------------------

angles = radar_angles(len(conditions))

plt.figure(figsize=(IEEE_COL_WIDTH, 3.25))
ax = plt.subplot(111, polar=True)

for i, model in enumerate(models):
    values = close_values(M_flip[i])
    ax.plot(
        angles,
        values,
        linewidth=1.7,
        color=MODEL_COLORS[i % len(MODEL_COLORS)],
        label=short_model_name(model),
    )
    ax.fill(
        angles,
        values,
        color=MODEL_COLORS[i % len(MODEL_COLORS)],
        alpha=0.06,
    )

ax.set_xticks(angles[:-1])
ax.set_xticklabels(condition_labels, fontsize=7.5)
ax.set_ylim(0, max(0.20, float(M_flip.max()) * 1.15))
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.tick_params(axis="y", labelsize=7)
ax.set_title("Sensitivity profile", pad=14)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=True, ncol=2, fontsize=6.5)
ax.grid(alpha=0.35)
savefig("fig5_radar_flip_by_model")


# ---------------------------------------------------------------------
# Fig 6 - spider/radar plot: average downgrade vs over-escalation
# Use for slides or appendix, not main paper.
# ---------------------------------------------------------------------

plt.figure(figsize=(IEEE_COL_WIDTH, 3.05))
ax = plt.subplot(111, polar=True)

ax.plot(
    angles,
    close_values(avg_down),
    linewidth=2.0,
    color=DOWNGRADE_COLOR,
    label="safety downgrade",
)
ax.fill(
    angles,
    close_values(avg_down),
    color=DOWNGRADE_COLOR,
    alpha=0.10,
)

ax.plot(
    angles,
    close_values(avg_up),
    linewidth=2.0,
    color=OVERESC_COLOR,
    label="over-escalation",
)
ax.fill(
    angles,
    close_values(avg_up),
    color=OVERESC_COLOR,
    alpha=0.10,
)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(condition_labels, fontsize=7.5)
ax.set_ylim(0, max(0.08, float(max(avg_down.max(), avg_up.max())) * 1.25))
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.tick_params(axis="y", labelsize=7)
ax.set_title("Instability direction", pad=14)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=True, ncol=1, fontsize=7)
ax.grid(alpha=0.35)
savefig("fig6_radar_harm_by_condition")


print("Done. All figures were written to the figures/ directory.")
