"""Paper-style figures (learning curves, correlation heatmap, backbone bars)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import (
    BACKBONE_TICKS,
    BACKBONES,
    METHOD_COLOR,
    METHOD_DISPLAY,
    METHOD_MARKER,
    METHODS,
    SAMPLE_SIZES,
)

sns.set_theme(style="darkgrid", font_scale=1.05)


# ------------------------------------------------------- learning curves
def _errorband(ax, xs, means, cis, color, marker, label):
    xs = np.asarray(xs, dtype=float)
    means = np.asarray(means, dtype=float)
    cis = np.asarray(cis, dtype=float)
    ax.errorbar(xs, means, yerr=cis, fmt=f"-{marker}", color=color,
                ecolor=color, elinewidth=1.6, capsize=4, capthick=1.6,
                markersize=7, markeredgewidth=1.6, linewidth=2.0,
                label=f"{label} (mean)" if "(" not in label else label,
                zorder=3)
    ax.fill_between(xs, means - cis, means + cis, color=color, alpha=0.18,
                    label=f"{label} 95% CI", zorder=2)


def _style_curve_ax(ax, title):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Number of patients", fontsize=11)
    ax.set_ylabel("Accuracy (Accuracy)", fontsize=11)
    ax.set_xlim(30, 1030)
    ax.set_xticks(SAMPLE_SIZES)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(0.55, 0.91)
    ax.set_yticks(np.arange(0.55, 0.911, 0.05))
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)


def plot_learning_curves(means_df, path):
    """2x2 figure: per-method average chart with error bars + comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.tight_layout(pad=3.0)
    for ax, method in zip([axes[0, 0], axes[0, 1], axes[1, 0]], METHODS):
        sub = means_df[means_df.method == method].sort_values("n")
        _errorband(ax, sub["n"], sub["accuracy_mean"], sub["accuracy_ci"],
                   METHOD_COLOR[method], METHOD_MARKER[method],
                   METHOD_DISPLAY[method])
        _style_curve_ax(ax, f"Average chart with error bars- {METHOD_DISPLAY[method]}")
    ax = axes[1, 1]
    for method in METHODS:
        sub = means_df[means_df.method == method].sort_values("n")
        _errorband(ax, sub["n"], sub["accuracy_mean"], sub["accuracy_ci"],
                   METHOD_COLOR[method], METHOD_MARKER[method],
                   METHOD_DISPLAY[method])
    _style_curve_ax(ax, "Comparison of all methods with error bars(95% CI)")
    # comparison panel uses short legend labels (no "(mean)" suffix)
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color=METHOD_COLOR[m],
                             marker=METHOD_MARKER[m], linestyle="-",
                             label=METHOD_DISPLAY[m]) for m in METHODS]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
              framealpha=0.95)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------- correlation heatmap
def plot_correlation_heatmap(corr_df, path):
    order = ["Patients", "SVM", "Ensemble", "FCLF"]
    disp = {"Patients": "Patients", "SVM": "SVM",
            "Ensemble": "Ensemble", "FCLF": "FCLF (Proposed)"}
    mat = corr_df.loc[order, order]
    fig, ax = plt.subplots(figsize=(10.5, 7))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="coolwarm", vmin=0.76, vmax=1.0,
                square=False, linewidths=1.0, linecolor="white",
                annot_kws={"fontsize": 11}, ax=ax,
                cbar_kws={"label": "Correlation coefficient"})
    ax.set_title("Heatmap of correlation between variables", fontsize=14, pad=12)
    ax.set_xticklabels([disp[c] for c in order], fontsize=11)
    ax.set_yticklabels([disp[c] for c in order], fontsize=11, rotation=90)
    ax.tick_params(axis="x", labelrotation=0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------- backbone group bars
def plot_backbone_comparison(backbone_df, path):
    df = backbone_df.set_index("backbone").loc[BACKBONES].reset_index()
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = {"accuracy": "Accuracy", "precision": "Precision",
              "recall": "Recall", "f1": "F1-Score"}
    colors = {"accuracy": "#1f77b4", "precision": "#ff7f0e",
              "recall": "#2ca02c", "f1": "#d62728"}
    x = np.arange(len(BACKBONES))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 7))
    for j, m in enumerate(metrics):
        ax.bar(x + (j - 1.5) * width, df[m], width=width,
               color=colors[m], label=labels[m], zorder=3, edgecolor="white")
    ax.set_title("Comparison of group-bar linguistic models", fontsize=14, pad=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(BACKBONE_TICKS, rotation=15, fontsize=11)
    ax.set_ylim(0.78, 0.95)
    ax.set_yticks(np.arange(0.78, 0.951, 0.02))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.995), fontsize=10,
              framealpha=0.95)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
