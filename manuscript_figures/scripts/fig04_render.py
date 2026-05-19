"""Figure 4 (DanioDecima MS): Developmental timepoint enrichment in model
performance extremes.

Two panels (bars per developmental timepoint, grouped by model type):
  A. Enrichment of the 50 poorest-performing pseudobulks (relative to background)
  B. Enrichment of the 50 best-performing pseudobulks (relative to background)

Source: 01_evaluate_celltypes.ipynb L3334 (`analyze_timepoint_enrichment_all_models`).
Publication style applied via figure_helpers.apply_style().
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    MODEL_COLORS, MODEL_ORDER, TIMEPOINT_ORDER, aggregate_by_model_type,
)

OUT_DIR = _THIS.parent / "figures"
OUT_STEM = OUT_DIR / "fig04_timepoint_enrichment"


def compute_enrichment(aggregated_data, kind):
    """kind = 'poorest' or 'best'. Returns DataFrame with per-(model,timepoint) enrichment."""
    rows = []
    for model_type, data in aggregated_data.items():
        combined_df = data["combined_df"]
        subset = data[kind]
        existing_tps = [tp for tp in TIMEPOINT_ORDER if tp in combined_df["timepoint"].unique()]
        total_sub = len(subset)
        total_all = len(combined_df)
        for tp in existing_tps:
            sub_count = int((subset["timepoint"] == tp).sum())
            all_count = int((combined_df["timepoint"] == tp).sum())
            sub_pct = (sub_count / total_sub * 100) if total_sub else 0
            all_pct = (all_count / total_all * 100) if total_all else 0
            enrich = (sub_pct / all_pct) if all_pct > 0 else 0
            rows.append({
                "model": model_type, "timepoint": tp,
                "count": sub_count, "enrichment": enrich,
            })
    return pd.DataFrame(rows)


def plot_panel(ax, df, models_present, timepoints_present, ylabel, title,
               y_cap, count_threshold):
    x = np.arange(len(timepoints_present))
    width = 0.2
    n_replicates_lookup = {m: agg[m]["n_replicates"] for m in models_present}
    for i, model in enumerate(MODEL_ORDER):
        if model not in models_present:
            continue
        model_df = df[df["model"] == model]
        enrichments, counts = [], []
        for tp in timepoints_present:
            row = model_df[model_df["timepoint"] == tp]
            if len(row) > 0:
                enrichments.append(row["enrichment"].iloc[0])
                counts.append(int(row["count"].iloc[0]))
            else:
                enrichments.append(0)
                counts.append(0)
        bars = ax.bar(
            x + i * width, enrichments, width,
            label=f"{model} (n={n_replicates_lookup[model]})",
            color=MODEL_COLORS[model], alpha=0.85,
        )
        for bar, count, enrich in zip(bars, counts, enrichments):
            if enrich > 0.3 and count > 0:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h * 0.9 if h > 0.5 else h + 0.05,
                    f"{count}",
                    ha="center",
                    va="center" if h > 0.5 else "bottom",
                    color="white" if h > 0.5 else "black",
                )
    ax.set_xlabel("developmental timepoint")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(timepoints_present, rotation=45, ha="right")
    ax.axhline(y=1, color="black", linestyle="--", alpha=0.6, linewidth=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    y_max = min(df["enrichment"].max() * 1.1, y_cap)
    ax.set_ylim(0, y_max)


def main():
    global agg
    agg = aggregate_by_model_type(n_performers=50)
    poor_df = compute_enrichment(agg, "poorest")
    best_df = compute_enrichment(agg, "best")

    models_present = [m for m in MODEL_ORDER if m in agg]
    timepoints_present = [
        tp for tp in TIMEPOINT_ORDER if tp in poor_df["timepoint"].unique()
    ]

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, wspace=0.45, top=0.82, bottom=0.18, left=0.07, right=0.88)
    ax_poor = fig.add_subplot(gs[0, 0])
    ax_best = fig.add_subplot(gs[0, 1])

    plot_panel(
        ax_poor, poor_df, models_present, timepoints_present,
        ylabel="enrichment in poor performers",
        title="A. poor performer timepoint enrichment",
        y_cap=4.0, count_threshold=0,
    )
    plot_panel(
        ax_best, best_df, models_present, timepoints_present,
        ylabel="enrichment in best performers",
        title="B. best performer timepoint enrichment",
        y_cap=6.0, count_threshold=0,
    )

    fig.suptitle(
        "developmental timepoint enrichment in model performance extremes",
        y=0.97,
    )
    save_figure(fig, OUT_STEM, dpi=300, verify=True, halt_on_fail=False)


if __name__ == "__main__":
    main()
