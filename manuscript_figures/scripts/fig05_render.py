"""Figure 5 (DanioDecima MS): Developmental performance patterns — absolute
performance, lift over Random, baseline-vs-improvement, summary table.

Five panels:
  A. Mean Pearson by developmental timepoint, per model
  B. Lift over Random Init by timepoint, per pretrained model
  C. Random Init baseline by timepoint (with "room for improvement")
  D. Scatter: Random baseline vs lift (with linregress fit + corr)
  E. Developmental-stage summary table

Source: 01_evaluate_celltypes.ipynb L4344 (`analyze_developmental_performance_patterns_fixed`).
Publication style applied via figure_helpers.apply_style().
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_FIGSTYLE_DIR = Path(
    "/home/yang-joon.kim/.claude/plugins/cache/yangjun9095-plugins/"
    "figure-style/1.0.0/scripts"
)
sys.path.insert(0, str(_FIGSTYLE_DIR))
from figure_helpers import apply_style, save_figure  # noqa: E402

apply_style()

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
from _fig04_05_loader import (  # noqa: E402
    DEV_STAGES, MODEL_COLORS, MODEL_ORDER, TIMEPOINT_ORDER,
    aggregate_by_model_type,
)

OUT_DIR = _THIS.parent / "figures"
OUT_STEM = OUT_DIR / "fig05_developmental_performance"


def build_timepoint_stats(aggregated_data):
    rows = []
    for tp in TIMEPOINT_ORDER:
        for model, data in aggregated_data.items():
            if model not in MODEL_ORDER:
                continue
            tp_data = data["combined_df"]
            tp_data = tp_data[tp_data["timepoint"] == tp]["test_pearson"].dropna()
            if len(tp_data) == 0:
                continue
            stage = "Unknown"
            for stage_name, tps in DEV_STAGES.items():
                if tp in tps:
                    stage = stage_name
                    break
            rows.append({
                "timepoint": tp, "model": model, "stage": stage,
                "mean_pearson": tp_data.mean(),
                "q75_pearson": tp_data.quantile(0.75),
                "q95_pearson": tp_data.quantile(0.95),
                "n_samples": len(tp_data),
            })
    return pd.DataFrame(rows)


def build_lift_stats(tp_stats_df):
    rows = []
    for tp in TIMEPOINT_ORDER:
        sub = tp_stats_df[tp_stats_df["timepoint"] == tp]
        random_row = sub[sub["model"] == "Random Init"]
        if len(random_row) == 0:
            continue
        random_mean = random_row["mean_pearson"].iloc[0]
        stage = random_row["stage"].iloc[0]
        for model in ["Mouse-Borzoi", "Human-Borzoi", "Human-Decima"]:
            model_row = sub[sub["model"] == model]
            if len(model_row) == 0:
                continue
            mm = model_row["mean_pearson"].iloc[0]
            rows.append({
                "timepoint": tp, "model": model, "stage": stage,
                "mean_lift": mm - random_mean,
                "random_baseline": random_mean,
                "pretrained_performance": mm,
            })
    return pd.DataFrame(rows)


def plot_panel_A(ax, tp_stats_df, timepoints_present):
    x = np.arange(len(timepoints_present))
    width = 0.2
    for i, model in enumerate(MODEL_ORDER):
        means = []
        for tp in timepoints_present:
            row = tp_stats_df[(tp_stats_df["model"] == model) & (tp_stats_df["timepoint"] == tp)]
            means.append(row["mean_pearson"].iloc[0] if len(row) > 0 else 0)
        ax.bar(x + i * width, means, width, label=model,
               color=MODEL_COLORS[model], alpha=0.85)
        if means and max(means) > 0.7:
            idx = int(np.argmax(means))
            ax.text(x[idx] + i * width, max(means) + 0.02, f"{max(means):.2f}",
                    ha="center", va="bottom", fontweight="bold")
    ax.set_xlabel("developmental timepoint")
    ax.set_ylabel("mean Pearson correlation")
    ax.set_title("A. absolute performance by timepoint")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(timepoints_present, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 0.95)


def plot_panel_B(ax, lift_df, timepoints_present):
    x = np.arange(len(timepoints_present))
    width = 0.25
    pretrained = ["Mouse-Borzoi", "Human-Borzoi", "Human-Decima"]
    for i, model in enumerate(pretrained):
        lifts = []
        for tp in timepoints_present:
            row = lift_df[(lift_df["model"] == model) & (lift_df["timepoint"] == tp)]
            lifts.append(row["mean_lift"].iloc[0] if len(row) > 0 else 0)
        ax.bar(x + i * width, lifts, width, label=model,
               color=MODEL_COLORS[model], alpha=0.85)
        if lifts and max(lifts) > 0.15:
            idx = int(np.argmax(lifts))
            ax.text(x[idx] + i * width, max(lifts) + 0.01, f"{max(lifts):.2f}",
                    ha="center", va="bottom", fontweight="bold")
    ax.set_xlabel("developmental timepoint")
    ax.set_ylabel("lift over Random Init")
    ax.set_title("B. improvement over Random by timepoint")
    ax.set_xticks(x + width)
    ax.set_xticklabels(timepoints_present, rotation=45, ha="right")
    ax.legend()
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=0.6)


def plot_panel_C(ax, tp_stats_df, timepoints_present):
    random_means = []
    for tp in timepoints_present:
        row = tp_stats_df[(tp_stats_df["model"] == "Random Init") & (tp_stats_df["timepoint"] == tp)]
        random_means.append(row["mean_pearson"].iloc[0] if len(row) > 0 else 0)
    ax.bar(timepoints_present, random_means, color=MODEL_COLORS["Random Init"], alpha=0.8)
    for i, (tp, baseline) in enumerate(zip(timepoints_present, random_means)):
        room = 1.0 - baseline
        if room > 0.3:
            ax.text(i, baseline + 0.02, f"room\n{room:.2f}",
                    ha="center", va="bottom")
    ax.set_xlabel("developmental timepoint")
    ax.set_ylabel("Random Init mean Pearson")
    ax.set_title("C. Random Init baseline performance")
    ax.tick_params(axis="x", rotation=45)


def plot_panel_D(ax, lift_df):
    pretrained = ["Mouse-Borzoi", "Human-Borzoi", "Human-Decima"]
    for model in pretrained:
        sub = lift_df[lift_df["model"] == model]
        ax.scatter(sub["random_baseline"], sub["mean_lift"],
                   label=model, color=MODEL_COLORS[model], alpha=0.8, s=30)
        for _, row in sub.iterrows():
            if row["mean_lift"] > 0.15 or row["random_baseline"] < 0.6:
                ax.annotate(row["timepoint"],
                            (row["random_baseline"], row["mean_lift"]),
                            xytext=(2, 2), textcoords="offset points")
    bx = lift_df["random_baseline"].values
    by = lift_df["mean_lift"].values
    slope, intercept, r_value, p_value, _ = stats.linregress(bx, by)
    line_x = np.linspace(bx.min(), bx.max(), 100)
    ax.plot(line_x, slope * line_x + intercept, "k--", alpha=0.5, linewidth=0.6)
    ax.text(0.05, 0.95, f"r = {r_value:.2f}\np = {p_value:.3f}",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("Random Init baseline")
    ax.set_ylabel("lift over Random Init")
    ax.set_title("D. baseline vs improvement")
    ax.legend(loc="lower right")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=0.6)


def plot_panel_E(ax, tp_stats_df, lift_df):
    rows = []
    for stage_name, stage_tps in DEV_STAGES.items():
        stage_random = tp_stats_df[
            tp_stats_df["timepoint"].isin(stage_tps)
            & (tp_stats_df["model"] == "Random Init")
        ]
        stage_pretrained = tp_stats_df[
            tp_stats_df["timepoint"].isin(stage_tps)
            & (tp_stats_df["model"] != "Random Init")
        ]
        stage_lifts = lift_df[lift_df["timepoint"].isin(stage_tps)]
        best_idx = stage_pretrained["mean_pearson"].idxmax()
        rows.append([
            stage_name,
            f"{stage_random['mean_pearson'].mean():.3f}",
            f"{stage_pretrained['mean_pearson'].max():.3f}",
            f"{stage_lifts['mean_lift'].mean():.3f}",
            f"{stage_lifts['mean_lift'].max():.3f}",
            stage_pretrained.loc[best_idx, "model"],
        ])
    table = ax.table(
        cellText=rows,
        colLabels=["developmental stage", "Random\nbaseline", "pretrained\nbest",
                   "average\nlift", "maximum\nlift", "best\nmodel"],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.0, 1.8)
    ax.set_title("E. developmental stage summary")
    ax.axis("off")
    for i, row in enumerate(rows):
        model = row[5]
        if model in MODEL_COLORS:
            cell = table[(i + 1, 5)]
            cell.set_facecolor(MODEL_COLORS[model])
            cell.set_alpha(0.3)


def main():
    agg = aggregate_by_model_type(n_performers=50)
    tp_stats_df = build_timepoint_stats(agg)
    lift_df = build_lift_stats(tp_stats_df)

    timepoints_present = [tp for tp in TIMEPOINT_ORDER if tp in tp_stats_df["timepoint"].unique()]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1, 1, 0.55],
        hspace=0.55, wspace=0.28,
        top=0.92, bottom=0.06, left=0.07, right=0.97,
    )
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])
    ax_E = fig.add_subplot(gs[2, :])

    plot_panel_A(ax_A, tp_stats_df, timepoints_present)
    plot_panel_B(ax_B, lift_df, timepoints_present)
    plot_panel_C(ax_C, tp_stats_df, timepoints_present)
    plot_panel_D(ax_D, lift_df)
    plot_panel_E(ax_E, tp_stats_df, lift_df)

    fig.suptitle(
        "developmental performance patterns: absolute performance vs improvement",
        y=0.96,
    )
    save_figure(fig, OUT_STEM, dpi=300, verify=True, halt_on_fail=False)


if __name__ == "__main__":
    main()
