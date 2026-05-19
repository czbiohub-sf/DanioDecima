"""
Merged Figure 9 + Figure 10 (DanioDecima MS) from designs_clustering_analysis.ipynb.

Cell 3 of the source notebook already combines a column dendrogram (top) with a
reordered heatmap (bottom). Cell 5 then re-draws the same dendrogram alone,
this time with cluster coloring via `color_threshold`. Since the dendrogram in
cell 5 carries no extra information beyond cell 3, this script renders one
merged figure that takes cell 3's structure and borrows cell 5's cluster
coloring.

Layout (option A — minimal merge):
  Top:    cell-type dendrogram, colored by clusters
  Bottom: heatmap (genes × 25 cell-types), columns reordered by clustering,
          z-scored per gene (row-wise)

Run under the gReLu conda env (with conda's libstdc++ on LD_LIBRARY_PATH).
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

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
OUT_STEM = OUT_DIR / "fig9_10_celltype_clustering_rev6"

CSV_PATH = (
    "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/design/"
    "analysis_25ct_20250619/comprehensive_summary_ism_90/"
    "motif_occurrence_by_celltype_combined_normalized.csv"
)


def load_zscored_matrix(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={df.columns[0]: "Gene"}).set_index("Gene")
    expr = df.drop(columns="Total")
    expr_z = (
        expr.sub(expr.mean(axis=1), axis=0)
        .div(expr.std(axis=1).replace(0, 1), axis=0)
    )
    return expr_z


def build_column_linkage(expr_z):
    dist_cols = pdist(expr_z.T, metric="correlation")
    Z = linkage(dist_cols, method="average")
    return Z


def build_row_linkage(expr_z):
    # Euclidean on z-scored rows ranks identically to correlation distance
    # (euclidean^2 = 2n(1-corr) for unit-variance vectors), and avoids NaNs
    # from any zero-variance rows that get z-scored to all zeros.
    dist_rows = pdist(expr_z.values, metric="euclidean")
    Z = linkage(dist_rows, method="average")
    return Z


def render_merged(expr_z, Z, Z_rows=None, figsize=(14, 11)):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[40, 1],
        height_ratios=[1, 3],
        hspace=0.05, wspace=0.04,
        left=0.10, right=0.95, top=0.93, bottom=0.20,
    )

    ax_dendro = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_cbar = fig.add_subplot(gs[1, 1])

    color_threshold = 0.85 * Z[:, 2].max()
    dendro = dendrogram(
        Z,
        labels=list(expr_z.columns),
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=color_threshold,
        ax=ax_dendro,
        above_threshold_color="#888888",
    )
    ax_dendro.set_ylabel("1 - corr", fontsize=11)
    ax_dendro.tick_params(axis="x", labelbottom=False, bottom=False)
    for spine in ("top", "right"):
        ax_dendro.spines[spine].set_visible(False)

    ordered_cols = dendro["leaves"]
    if Z_rows is not None:
        row_dendro = dendrogram(Z_rows, no_plot=True)
        ordered_rows = row_dendro["leaves"]
        expr_reordered = expr_z.iloc[ordered_rows, ordered_cols]
    else:
        expr_reordered = expr_z.iloc[:, ordered_cols]

    vmin, vmax = -1, 4
    data_min = float(np.nanmin(expr_reordered.values))
    data_max = float(np.nanmax(expr_reordered.values))
    print(f"Data z-score range: [{data_min:.2f}, {data_max:.2f}]; clipped to [{vmin}, {vmax}]")
    im = ax_heat.imshow(
        expr_reordered.values,
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_heat.set_xticks(range(expr_reordered.shape[1]))
    ax_heat.set_xticklabels(expr_reordered.columns, rotation=90, fontsize=10)
    ax_heat.set_yticks([])
    ax_heat.set_ylabel(f"TF motifs (n={expr_reordered.shape[0]})", fontsize=11)

    cbar = fig.colorbar(im, cax=ax_cbar, extend="both", extendrect=True)
    cbar.set_label("TF-motif z-score", fontsize=11)
    cbar.set_ticks([-1, 0, 1, 2, 3, 4])
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Cell-type similarity (correlation distance, average linkage)",
        fontsize=14, fontweight="bold", y=0.97,
    )
    return fig


def main():
    expr_z = load_zscored_matrix(CSV_PATH)
    print(f"Loaded matrix: {expr_z.shape[0]} genes x {expr_z.shape[1]} cell-types")
    Z = build_column_linkage(expr_z)
    Z_rows = build_row_linkage(expr_z)
    fig = render_merged(expr_z, Z, Z_rows=Z_rows)
    save_figure(fig, OUT_STEM, dpi=300, verify=True, halt_on_fail=False)


if __name__ == "__main__":
    main()
