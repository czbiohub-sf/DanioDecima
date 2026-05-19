"""
Re-render Figure 3 (DanioDecima MS) from 01_evaluate_celltypes.ipynb.

Changes vs. the original cell at L845-L1019:
  - height_ratios:  [2.5, 2.5, 1.2]  ->  [2.5, 2.5, 2.5]   (taller bottom boxplot row)
  - figsize:        (24, 14)          ->  (24, 16)
  - rcParams set so the saved PDF embeds TrueType fonts (editable in Illustrator/Inkscape)
  - Saves both .pdf and .png to figures/ instead of plt.show()

Run with the gReLu conda env.
"""

import sys
from pathlib import Path

import anndata
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

_FIGSTYLE_DIR = Path(
    "/home/yang-joon.kim/.claude/plugins/cache/yangjun9095-plugins/"
    "figure-style/1.0.0/scripts"
)
if _FIGSTYLE_DIR.exists():
    sys.path.insert(0, str(_FIGSTYLE_DIR))
from figure_helpers import save_figure  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_STEM = OUT_DIR / "fig3_model_comparison_rev3"

BASE = "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments"
EXP1 = f"{BASE}/decima_experiments_20250617_225114"
EXP2 = f"{BASE}/decima_experiments_20250618_111138"

MODEL_PATHS = [
    f"{EXP1}/random_lr3e-06_seed42/version_0/data_out_decima_random_lr3e-06_seed42_20250619_Random_0.h5ad",
    f"{EXP1}/random_lr3e-06_seed43/version_0/data_out_decima_random_lr3e-06_seed43_20250619_Random_1.h5ad",
    f"{EXP1}/random_lr3e-06_seed44/version_0/data_out_decima_random_lr3e-06_seed44_20250619_Random_2.h5ad",
    f"{EXP1}/random_lr3e-06_seed45/version_0/data_out_decima_random_lr3e-06_seed45_20250619_Random_3.h5ad",
    f"{EXP1}/pretrained_wandb-human_rep0_lr3e-05_seed42/version_0/data_out_decima_pretrained_wandb-human_rep0_lr3e-05_seed42_20250619_Human_Borzoi_0.h5ad",
    f"{EXP1}/pretrained_wandb-human_rep1_lr3e-05_seed42/version_0/data_out_decima_pretrained_wandb-human_rep1_lr3e-05_seed42_20250619_Human_Borzoi_1.h5ad",
    f"{EXP1}/pretrained_wandb-human_rep2_lr3e-05_seed42/version_0/data_out_decima_pretrained_wandb-human_rep2_lr3e-05_seed42_20250619_Human_Borzoi_2.h5ad",
    f"{EXP1}/pretrained_wandb-human_rep3_lr3e-05_seed42/version_0/data_out_decima_pretrained_wandb-human_rep3_lr3e-05_seed42_20250619_Human_Borzoi_3.h5ad",
    f"{EXP1}/pretrained_wandb-mouse_rep0_lr3e-05_seed42/version_0/data_out_decima_pretrained_wandb-mouse_rep0_lr3e-05_seed42_20250619_Mouse_Borzoi_0.h5ad",
    f"{EXP1}/pretrained_wandb-mouse_rep1_lr3e-05_seed42/version_0/data_out_decima_pretrained_wandb-mouse_rep1_lr3e-05_seed42_20250619_Mouse_Borzoi_1.h5ad",
    f"{EXP1}/pretrained_wandb-mouse_rep2_lr3e-05_seed42/version_0/data_out_decima_pretrained_wandb-mouse_rep2_lr3e-05_seed42_20250619_Mouse_Borzoi_2.h5ad",
    f"{EXP1}/pretrained_wandb-mouse_rep3_lr3e-05_seed42/version_0/data_out_decima_pretrained_wandb-mouse_rep3_lr3e-05_seed42_20250619_Mouse_Borzoi_3.h5ad",
    f"{EXP2}/pretrained_decima-human_rep0_lr3e-05_seed42/version_0/data_out_decima_pretrained_decima-human_rep0_lr3e-05_seed42_20250619_Human_Decima_0.h5ad",
    f"{EXP2}/pretrained_decima-human_rep1_lr3e-05_seed42/version_0/data_out_decima_pretrained_decima-human_rep1_lr3e-05_seed42_20250619_Human_Decima_1.h5ad",
    f"{EXP2}/pretrained_decima-human_rep2_lr3e-05_seed42/version_0/data_out_decima_pretrained_decima-human_rep2_lr3e-05_seed42_20250619_Human_Decima_2.h5ad",
    f"{EXP2}/pretrained_decima-human_rep3_lr3e-05_seed42/version_0/data_out_decima_pretrained_decima-human_rep3_lr3e-05_seed42_20250619_Human_Decima_3.h5ad",
]

MODEL_NAMES = [
    "Random_0", "Random_1", "Random_2", "Random_3",
    "Human_Borzoi_0", "Human_Borzoi_1", "Human_Borzoi_2", "Human_Borzoi_3",
    "Mouse_Borzoi_0", "Mouse_Borzoi_1", "Mouse_Borzoi_2", "Mouse_Borzoi_3",
    "Human_Decima_0", "Human_Decima_1", "Human_Decima_2", "Human_Decima_3",
]


def load_multiple_models(model_paths, model_names):
    obs_chunks, var_chunks = [], []
    for path, name in zip(model_paths, model_names):
        ad_temp = anndata.read_h5ad(path)
        obs_temp = ad_temp.obs.copy()
        obs_temp["model_type"] = name
        obs_temp["model_file"] = path.split("/")[-1]
        obs_chunks.append(obs_temp)
        var_temp = ad_temp.var.copy()
        var_temp["model_type"] = name
        var_temp["model_file"] = path.split("/")[-1]
        var_chunks.append(var_temp)
        print(f"Loaded {name}: {len(obs_temp)} pseudobulks, {len(var_temp)} genes")
    return (
        pd.concat(obs_chunks, ignore_index=True),
        pd.concat(var_chunks, ignore_index=True),
    )


def pairwise_mannwhitney(data_by_group, group_order):
    """Pairwise Mann-Whitney U with Benjamini-Hochberg FDR correction.

    Returns a DataFrame with columns: group1, group2, p_value, p_corrected.
    """
    rows = []
    for i, g1 in enumerate(group_order):
        for j, g2 in enumerate(group_order):
            if i >= j:
                continue
            d1 = data_by_group.get(g1)
            d2 = data_by_group.get(g2)
            if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
                continue
            _, p = stats.mannwhitneyu(d1, d2, alternative="two-sided")
            rows.append({"group1": g1, "group2": g2, "p_value": p})
    df = pd.DataFrame(rows)
    if len(df) > 1:
        _, p_corr, _, _ = multipletests(df["p_value"], method="fdr_bh")
        df["p_corrected"] = p_corr
    elif len(df) == 1:
        df["p_corrected"] = df["p_value"]
    return df


def add_significance_brackets(ax, test_results, group_order, positions,
                              y_offset_start=0.05, bracket_step=0.08):
    """Stack significance brackets above an axes' existing y-range.

    Adapted from the notebook's add_significance_brackets (L1321). Uses
    `p_corrected` (BH-FDR) if present, else `p_value`. Only draws p<0.05.
    Brackets are ordered by group-distance: nearest pair first (lowest level),
    then increasing distance stacks above.
    """
    if len(test_results) == 0:
        return
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    pos_map = {g: positions[i] for i, g in enumerate(group_order)}
    p_col = "p_corrected" if "p_corrected" in test_results.columns else "p_value"

    comparisons = []
    for _, row in test_results.iterrows():
        g1, g2 = row["group1"], row["group2"]
        if g1 not in pos_map or g2 not in pos_map:
            continue
        p = row[p_col]
        if p >= 0.05:
            continue
        comparisons.append({
            "x1": pos_map[g1], "x2": pos_map[g2],
            "distance": abs(pos_map[g2] - pos_map[g1]),
            "p": p,
        })
    comparisons.sort(key=lambda c: c["distance"])

    bracket_height = y_range * 0.03
    for level, comp in enumerate(comparisons):
        p = comp["p"]
        if p < 0.001:
            sym = "***"
        elif p < 0.01:
            sym = "**"
        else:
            sym = "*"
        y = y_max + y_range * (y_offset_start + level * bracket_step)
        x1, x2 = comp["x1"], comp["x2"]
        ax.plot(
            [x1, x1, x2, x2],
            [y - bracket_height / 2, y, y, y - bracket_height / 2],
            "k-", linewidth=1,
        )
        ax.text((x1 + x2) / 2, y + bracket_height / 2, sym,
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    if comparisons:
        new_y_max = y_max + y_range * (
            y_offset_start + len(comparisons) * bracket_step + 0.05
        )
        ax.set_ylim(y_min, new_y_max)


def create_clear_model_comparison_improved(
    combined_obs, combined_var, dataset="test", figsize=(24, 20)
):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        4, 4, height_ratios=[2.5, 2.5, 2.0, 2.0], hspace=0.5, wspace=0.35
    )

    group_colors = {
        "Random": "#1f77b4",
        "Mouse_Borzoi": "#ff7f0e",
        "Human_Borzoi": "#2ca02c",
        "Human_Decima": "#d62728",
    }

    def get_model_group(model_type):
        if "Random" in model_type:
            return "Random"
        elif "Human_Borzoi" in model_type:
            return "Human_Borzoi"
        elif "Mouse_Borzoi" in model_type:
            return "Mouse_Borzoi"
        elif "Human_Decima" in model_type:
            return "Human_Decima"
        else:
            return "Other"

    combined_obs["model_group"] = combined_obs["model_type"].apply(get_model_group)
    combined_var["model_group"] = combined_var["model_type"].apply(get_model_group)
    group_order = ["Random", "Mouse_Borzoi", "Human_Borzoi", "Human_Decima"]

    col_name = f"{dataset}_pearson"
    if col_name in combined_obs.columns:
        for i, group in enumerate(group_order):
            ax = fig.add_subplot(gs[0, i])
            group_data = (
                combined_obs[combined_obs["model_group"] == group][col_name].dropna()
            )
            if len(group_data) > 0:
                ax.hist(
                    group_data, bins=25, alpha=0.7, color=group_colors[group],
                    density=True, edgecolor="black", linewidth=0.5,
                )
                mean_val = group_data.mean()
                median_val = group_data.median()
                ax.axvline(mean_val, color="red", linestyle="--", linewidth=2)
                ax.axvline(median_val, color="orange", linestyle=":", linewidth=2)
                ax.set_title(
                    f'{group.replace("_", " ")}\nPseudobulk Correlations (n={len(group_data)})',
                    fontweight="bold", fontsize=12, pad=15,
                )
                ax.set_xlabel("Pearson Correlation", fontsize=10)
                ax.set_ylabel("Density", fontsize=10)
                ax.legend(
                    [f"Mean: {mean_val:.3f}", f"Median: {median_val:.3f}"],
                    loc="upper left", fontsize=9, framealpha=0.9,
                )
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0.4, 0.85)
                ax.tick_params(axis="both", which="major", labelsize=9)

    dataset_var = combined_var[combined_var["dataset"] == dataset]
    if len(dataset_var) > 0:
        for i, group in enumerate(group_order):
            ax = fig.add_subplot(gs[1, i])
            group_data = (
                dataset_var[dataset_var["model_group"] == group]["pearson"].dropna()
            )
            if len(group_data) > 0:
                ax.hist(
                    group_data, bins=25, alpha=0.7, color=group_colors[group],
                    density=True, edgecolor="black", linewidth=0.5,
                )
                mean_val = group_data.mean()
                median_val = group_data.median()
                ax.axvline(mean_val, color="red", linestyle="--", linewidth=2)
                ax.axvline(median_val, color="orange", linestyle=":", linewidth=2)
                ax.set_title(
                    f'{group.replace("_", " ")}\nGene Correlations (n={len(group_data)})',
                    fontweight="bold", fontsize=12, pad=15,
                )
                ax.set_xlabel("Pearson Correlation", fontsize=10)
                ax.set_ylabel("Density", fontsize=10)
                ax.legend(
                    [f"Mean: {mean_val:.3f}", f"Median: {median_val:.3f}"],
                    loc="upper left", fontsize=9, framealpha=0.9,
                )
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-0.75, 1.0)
                ax.tick_params(axis="both", which="major", labelsize=9)

    pseudobulk_by_group = {}
    gene_by_group = {}
    for group in group_order:
        pb = combined_obs[combined_obs["model_group"] == group][col_name].dropna()
        if len(pb) > 0:
            pseudobulk_by_group[group] = pb
        gd = dataset_var[dataset_var["model_group"] == group]["pearson"].dropna()
        if len(gd) > 0:
            gene_by_group[group] = gd

    box_labels = [g.replace("_", "\n") for g in group_order]
    box_colors = [group_colors[g] for g in group_order]
    box_positions = list(range(1, len(group_order) + 1))

    def _draw_box_row(ax, data_by_group, title):
        data = [data_by_group.get(g, pd.Series(dtype=float)) for g in group_order]
        bp = ax.boxplot(
            data, tick_labels=box_labels, patch_artist=True, showfliers=False
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(title, fontweight="bold", fontsize=14, pad=18)
        ax.set_ylabel("Pearson Correlation", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        tests = pairwise_mannwhitney(data_by_group, group_order)
        add_significance_brackets(ax, tests, group_order, box_positions)
        return tests

    ax_pb = fig.add_subplot(gs[2, :])
    pb_tests = _draw_box_row(ax_pb, pseudobulk_by_group, "Pseudobulk Correlations")

    ax_gn = fig.add_subplot(gs[3, :])
    gn_tests = _draw_box_row(ax_gn, gene_by_group, "Gene Correlations")

    print("\nPseudobulk pairwise (BH-FDR corrected):")
    print(pb_tests.to_string(index=False))
    print("\nGene pairwise (BH-FDR corrected):")
    print(gn_tests.to_string(index=False))

    plt.suptitle(
        f"Model Performance Comparison - {dataset.upper()} Dataset",
        fontsize=18, fontweight="bold", y=0.96,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    return fig


def main():
    combined_obs, combined_var = load_multiple_models(MODEL_PATHS, MODEL_NAMES)
    fig = create_clear_model_comparison_improved(
        combined_obs, combined_var, dataset="test", figsize=(24, 20)
    )
    save_figure(fig, OUT_STEM, dpi=300, verify=True, halt_on_fail=False)


if __name__ == "__main__":
    main()
