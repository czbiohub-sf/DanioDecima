# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DanioCell Decima — Manuscript Figures
#
# Reproduces publication-quality figures equivalent to Figures 1–10 of the
# DanioDecima manuscript, adapted for DanioCell data.
#
# **Run with:** `module load anaconda && source activate gReLu`
#
# ## Figure mapping
# | Figure | Description | Source |
# |--------|-------------|--------|
# | 1 | Training curves (val_loss, val_pearson) | CSVLogger metrics |
# | 3 | Model comparison: histograms + boxplots | 01_predictions h5ad |
# | 4 | Poor/best performer enrichment by stage | 01_predictions h5ad |
# | 5 | Developmental performance patterns | 01_predictions h5ad |
# | 6 | Conservation vs expression confounding | Predictions + orthologs |
# | 7 | Attribution by genomic region | 03_attributions pkl |
# | 8 | CRE fold change by region | 03_attributions pkl |
# | 9 | Motif clustering heatmap | 05_modisco h5 |
# | 10 | Cell-type dendrogram | 05_modisco h5 |

# %%
import numpy as np
import pandas as pd
import anndata
import os
import sys
import pickle
import warnings
from glob import glob
from scipy.stats import mannwhitneyu, spearmanr, pearsonr
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib import cm
import seaborn as sns
import h5py

warnings.filterwarnings('ignore', category=FutureWarning)

# %% [markdown]
# ## Configuration

# %%
# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = "/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell"
PRED_DIR = os.path.join(BASE_DIR, "01_predictions")
EVAL_DIR = os.path.join(BASE_DIR, "02_evaluation")
ATTR_DIR = os.path.join(BASE_DIR, "03_attributions")
SPEC_DIR = os.path.join(BASE_DIR, "04_specificity")
MODISC_DIR = os.path.join(BASE_DIR, "05_modisco")
EXP_DIR = os.path.join(BASE_DIR, "experiments",
                        "daniocell_experiments_20260406_114453")
ORTHO_SRC = ("/hpc/mydata/mathias.voges/Projects/research/seq2fun/step/"
              "decima-applications-main/notebooks/5_specificity/"
              "zebrafish_human_orthologs.csv")

FIG_DIR = os.path.join(EVAL_DIR, "manuscript_figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Experiment names ───────────────────────────────────────────────────────────
PRETRAINED_NAMES = [
    "pretrained_decima-human_rep0_lr3e-05_seed42",
    "pretrained_decima-human_rep1_lr3e-05_seed42",
    "pretrained_decima-human_rep2_lr3e-05_seed42",
    "pretrained_decima-human_rep3_lr3e-05_seed42",
]
RANDOM_NAMES = [
    "random_lr3e-06_seed42",
    "random_lr3e-06_seed43",
    "random_lr3e-06_seed44",
    "random_lr3e-06_seed45",
]
ALL_NAMES = PRETRAINED_NAMES + RANDOM_NAMES

# ── Attribution model labels ───────────────────────────────────────────────────
ATTR_MODELS = {
    "pretrained_rep0": "pretrained",
    "pretrained_rep1": "pretrained",
    "pretrained_rep2": "pretrained",
    "pretrained_rep3": "pretrained",
    "random_seed42": "random",
}

# ── Cell types for specificity / modisco ───────────────────────────────────────
FOCAL_CELLTYPES = [
    "cardiac_muscle", "epidermis", "intestine", "liver", "motor_neurons",
    "neural_crest", "neurons", "notochord", "radial_glia", "somite",
]

# ── Metadata keys ──────────────────────────────────────────────────────────────
CELLTYPE_KEY = "identity.super"
STAGE_KEY = "stage.group"

# ── Stage group ordering (hpf) ────────────────────────────────────────────────
STAGE_ORDER = [
    "3-4", "5-6", "7-9", "10-11", "12-13", "14-17", "18-21",
    "22-23", "24-35", "36-47", "48-59", "60-71", "72-95", "120",
]

# ── Plot style ─────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
COLORS = {"pretrained": "#2166ac", "random": "#b2182b"}
DPI = 300

def save_fig(fig, name):
    """Save figure as both PDF and PNG."""
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"),
                    dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {name}")


# %% [markdown]
# ## Load prediction data (shared across Figs 3–6)

# %%
def load_predictions(pred_dir, exp_names):
    """Load prediction h5ad files into a dict keyed by experiment name."""
    ads = {}
    for name in exp_names:
        path = os.path.join(pred_dir, f"data_out_daniocell_{name}.h5ad")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}")
            continue
        ads[name] = anndata.read_h5ad(path)
        print(f"Loaded {name}: {ads[name].shape}")
    return ads

ads = load_predictions(PRED_DIR, ALL_NAMES)
print(f"\nLoaded {len(ads)} / {len(ALL_NAMES)} models")

# Quick sanity check
for name, ad in ads.items():
    test_r = ad.var.loc[ad.var.dataset == 'test', 'pearson'].mean()
    group = "pretrained" if "pretrained" in name else "random"
    print(f"  {name}: test Pearson = {test_r:.4f}  ({group})")

# %%
# Build master summary DataFrames for downstream figures
def build_gene_pearson_df(ads):
    """Per-gene Pearson for each experiment (test set only)."""
    rows = []
    for name, ad in ads.items():
        group = "pretrained" if "pretrained" in name else "random"
        test_mask = ad.var.dataset == "test"
        for gene, r in zip(ad.var_names[test_mask],
                           ad.var.loc[test_mask, "pearson"].values):
            rows.append({"exp": name, "group": group,
                         "gene": gene, "pearson": r})
    return pd.DataFrame(rows)

def build_track_pearson_df(ads):
    """Per-track Pearson for each experiment (test genes)."""
    rows = []
    for name, ad in ads.items():
        if "test_pearson" not in ad.obs.columns:
            continue
        group = "pretrained" if "pretrained" in name else "random"
        for idx in range(ad.n_obs):
            rows.append({
                "exp": name, "group": group,
                "track_pearson": ad.obs["test_pearson"].iloc[idx],
                CELLTYPE_KEY: ad.obs[CELLTYPE_KEY].iloc[idx],
                STAGE_KEY: str(ad.obs[STAGE_KEY].iloc[idx]).strip(),
            })
    return pd.DataFrame(rows)

df_gene = build_gene_pearson_df(ads)
df_track = build_track_pearson_df(ads)
print(f"Gene Pearson rows: {len(df_gene):,}")
print(f"Track Pearson rows: {len(df_track):,}")

# Mean Pearson per gene across replicates
ref_ad = next(iter(ads.values()))
test_genes = ref_ad.var_names[ref_ad.var.dataset == "test"].tolist()

gene_means = df_gene.groupby(["group", "gene"]).pearson.mean().reset_index()
gene_means_wide = gene_means.pivot(index="gene", columns="group",
                                    values="pearson")

# %% [markdown]
# ---
# ## Figure 1 — Training Curves
#
# Adapted from `results_celltype_models.ipynb`. Reads CSVLogger output.

# %%
def load_training_metrics(exp_dir, exp_names):
    """Load CSVLogger metrics.csv files for all experiments."""
    dfs = []
    for name in exp_names:
        path = os.path.join(exp_dir, name, "version_0", "metrics.csv")
        if not os.path.exists(path):
            print(f"WARNING: missing metrics for {name}")
            continue
        df = pd.read_csv(path)
        df["exp"] = name
        df["group"] = "pretrained" if "pretrained" in name else "random"
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

df_metrics = load_training_metrics(EXP_DIR, ALL_NAMES)
print(f"Training metrics rows: {len(df_metrics):,}")

# %%
# Figure 1: Two-panel — val_loss and val_pearson across epochs
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for metric, ax, ylabel in [
    ("val_loss", axes[0], "Validation loss"),
    ("val_pearson", axes[1], "Validation Pearson r"),
]:
    for name in ALL_NAMES:
        sub = df_metrics[df_metrics.exp == name].dropna(subset=[metric])
        if sub.empty:
            continue
        group = "pretrained" if "pretrained" in name else "random"
        alpha = 0.7 if group == "pretrained" else 0.5
        ls = "-" if group == "pretrained" else "--"
        ax.plot(sub.epoch, sub[metric], color=COLORS[group],
                alpha=alpha, linestyle=ls, linewidth=1.2)

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color=COLORS["pretrained"], lw=2, label="Pretrained"),
        Line2D([0], [0], color=COLORS["random"], lw=2, ls="--", label="Random"),
    ]
    ax.legend(handles=legend_lines, frameon=True)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)

fig.suptitle("DanioCell training dynamics (8 experiments)", fontsize=13, y=1.02)
fig.tight_layout()
save_fig(fig, "fig1_training_curves")

# %% [markdown]
# ---
# ## Figure 3 — Model Comparison: Histograms & Boxplots
#
# Adapted from `01_evaluate_celltypes.ipynb`. Three-row panel.
# Row 1: Pseudobulk/track Pearson histograms
# Row 2: Per-gene Pearson histograms
# Row 3: Boxplots comparing pretrained vs random

# %%
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# ── Panel A: Per-gene Pearson histograms ──────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
for group, color in COLORS.items():
    vals = df_gene.loc[df_gene.group == group, "pearson"]
    ax_a.hist(vals, bins=80, alpha=0.55, color=color, label=group, density=True)
ax_a.set_xlabel("Pearson r (per gene, test set)")
ax_a.set_ylabel("Density")
ax_a.set_title("A. Per-gene correlation")
ax_a.legend()

# ── Panel B: Per-track Pearson histograms ─────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
if len(df_track) > 0:
    for group, color in COLORS.items():
        vals = df_track.loc[df_track.group == group, "track_pearson"]
        if len(vals) == 0:
            continue
        ax_b.hist(vals, bins=80, alpha=0.55, color=color,
                  label=group, density=True)
    ax_b.set_xlabel("Pearson r (per track, test genes)")
    ax_b.set_ylabel("Density")
    ax_b.set_title("B. Per-track correlation")
    ax_b.legend()
else:
    ax_b.text(0.5, 0.5, "No per-track data", ha='center', va='center',
              transform=ax_b.transAxes)

# ── Panel C: Scatter — pretrained vs random per gene ──────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
if "pretrained" in gene_means_wide.columns and "random" in gene_means_wide.columns:
    x = gene_means_wide["random"].values
    y = gene_means_wide["pretrained"].values
    valid = np.isfinite(x) & np.isfinite(y)
    ax_c.scatter(x[valid], y[valid], s=3, alpha=0.3, color="grey")
    lims = [min(np.nanmin(x[valid]), np.nanmin(y[valid])) - 0.02,
            max(np.nanmax(x[valid]), np.nanmax(y[valid])) + 0.02]
    ax_c.plot(lims, lims, '--', color='black', linewidth=0.8, alpha=0.5)
    ax_c.set_xlabel("Random init (mean Pearson)")
    ax_c.set_ylabel("Pretrained (mean Pearson)")
    ax_c.set_title("C. Per-gene: pretrained vs random")
    # Fraction above diagonal
    frac_above = np.mean(y[valid] > x[valid])
    ax_c.text(0.05, 0.95, f"{frac_above:.0%} genes above\ndiagonal",
              transform=ax_c.transAxes, va='top', fontsize=9,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ── Panel D: Summary boxplot by group ─────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
exp_means = df_gene.groupby(["exp", "group"]).pearson.mean().reset_index()
sns.boxplot(data=exp_means, x="group", y="pearson",
            palette=COLORS, width=0.5, ax=ax_d)
sns.stripplot(data=exp_means, x="group", y="pearson",
              color="black", size=7, ax=ax_d, zorder=3)
ax_d.set_xlabel("")
ax_d.set_ylabel("Mean Pearson r (test genes)")
ax_d.set_title("D. Model group comparison")

# Mann-Whitney U test
pre_vals = exp_means.loc[exp_means.group == "pretrained", "pearson"].values
rnd_vals = exp_means.loc[exp_means.group == "random", "pearson"].values
if len(pre_vals) > 1 and len(rnd_vals) > 1:
    _, pval = mannwhitneyu(pre_vals, rnd_vals, alternative='greater')
    ax_d.text(0.5, 0.95, f"MWU p = {pval:.3f}", transform=ax_d.transAxes,
              ha='center', va='top', fontsize=9)

fig.suptitle("Figure 3: DanioCell model comparison", fontsize=14, y=1.02)
save_fig(fig, "fig3_model_comparison")

# %% [markdown]
# ---
# ## Figure 4 — Poor/Best Performer Enrichment by Developmental Stage
#
# Adapted from `01_evaluate_celltypes.ipynb`.
# For each stage.group, compute the fraction of tracks that fall in the
# bottom 10% (poor) or top 10% (best) of Pearson distributions.

# %%
if len(df_track) > 0:
    # Compute per-group thresholds (10th / 90th percentile)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for gi, (group, color) in enumerate(COLORS.items()):
        sub = df_track[df_track.group == group].copy()
        p10 = sub.track_pearson.quantile(0.10)
        p90 = sub.track_pearson.quantile(0.90)
        sub["category"] = "middle"
        sub.loc[sub.track_pearson <= p10, "category"] = "poor"
        sub.loc[sub.track_pearson >= p90, "category"] = "best"

        # Fraction per stage
        stage_fracs = []
        for stage in STAGE_ORDER:
            stage_sub = sub[sub[STAGE_KEY] == stage]
            n_total = len(stage_sub)
            if n_total == 0:
                continue
            n_poor = (stage_sub.category == "poor").sum()
            n_best = (stage_sub.category == "best").sum()
            stage_fracs.append({
                "stage": stage,
                "poor_frac": n_poor / n_total,
                "best_frac": n_best / n_total,
            })
        sf = pd.DataFrame(stage_fracs)

        ax = axes[gi]
        x = np.arange(len(sf))
        width = 0.35
        ax.bar(x - width/2, sf.poor_frac, width, color="#d73027",
               alpha=0.7, label="Poor (bottom 10%)")
        ax.bar(x + width/2, sf.best_frac, width, color="#4575b4",
               alpha=0.7, label="Best (top 10%)")
        ax.axhline(0.10, color='grey', linestyle='--', linewidth=0.8,
                    alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s} hpf" for s in sf.stage],
                           rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Fraction of tracks")
        ax.set_title(f"{group.capitalize()}")
        ax.legend(fontsize=8)

    fig.suptitle("Figure 4: Poor/best performer enrichment by developmental stage",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig4_timepoint_enrichment")
else:
    print("Skipping Fig 4: no per-track Pearson data")

# %% [markdown]
# ---
# ## Figure 5 — Developmental Performance Patterns
#
# Four panels:
# A. Absolute performance by stage (mean track Pearson)
# B. Performance lift (pretrained − random) by stage
# C. Lift distribution across stages
# D. Top cell types by pretrained advantage

# %%
if len(df_track) > 0:
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    # ── Panel A: Mean track Pearson by stage ──────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    for group, color in COLORS.items():
        sub = df_track[df_track.group == group]
        stage_stats = sub.groupby(STAGE_KEY).track_pearson.agg(["mean", "sem"])
        stage_stats = stage_stats.reindex(
            [s for s in STAGE_ORDER if s in stage_stats.index])
        ax_a.errorbar(range(len(stage_stats)), stage_stats["mean"],
                      yerr=stage_stats["sem"], marker='o', color=color,
                      label=group, capsize=3, linewidth=1.5, markersize=5)
    ax_a.set_xticks(range(len([s for s in STAGE_ORDER
                               if s in df_track[STAGE_KEY].values])))
    present_stages = [s for s in STAGE_ORDER if s in df_track[STAGE_KEY].values]
    ax_a.set_xticklabels([f"{s}" for s in present_stages],
                         rotation=45, ha='right', fontsize=8)
    ax_a.set_xlabel("Stage group (hpf)")
    ax_a.set_ylabel("Mean Pearson r")
    ax_a.set_title("A. Absolute performance by stage")
    ax_a.legend()

    # ── Panel B: Performance lift by stage ────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    # Mean across replicates per stage/group
    stage_group_means = (df_track.groupby([STAGE_KEY, "group"])
                         .track_pearson.mean().unstack("group"))
    stage_group_means = stage_group_means.reindex(
        [s for s in STAGE_ORDER if s in stage_group_means.index])
    if "pretrained" in stage_group_means.columns and "random" in stage_group_means.columns:
        lift = stage_group_means["pretrained"] - stage_group_means["random"]
        bar_colors = [COLORS["pretrained"] if v >= 0 else COLORS["random"]
                      for v in lift.values]
        ax_b.bar(range(len(lift)), lift.values, color=bar_colors, alpha=0.7)
        ax_b.axhline(0, color='black', linewidth=0.8)
        ax_b.set_xticks(range(len(lift)))
        ax_b.set_xticklabels([f"{s}" for s in lift.index],
                             rotation=45, ha='right', fontsize=8)
        ax_b.set_xlabel("Stage group (hpf)")
        ax_b.set_ylabel("Pearson lift (pretrained - random)")
        ax_b.set_title("B. Pretrained advantage by stage")

    # ── Panel C: Lift distribution (per track) ────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    # Pair tracks across model groups (same cell type / stage)
    pre_means = (df_track[df_track.group == "pretrained"]
                 .groupby([CELLTYPE_KEY, STAGE_KEY]).track_pearson.mean()
                 .rename("pretrained"))
    rnd_means = (df_track[df_track.group == "random"]
                 .groupby([CELLTYPE_KEY, STAGE_KEY]).track_pearson.mean()
                 .rename("random"))
    paired = pd.concat([pre_means, rnd_means], axis=1).dropna()
    paired["lift"] = paired["pretrained"] - paired["random"]
    ax_c.hist(paired["lift"], bins=60, color=COLORS["pretrained"],
              alpha=0.7, edgecolor='white')
    ax_c.axvline(0, color='black', linewidth=1)
    ax_c.axvline(paired["lift"].median(), color='red', linewidth=1.5,
                 linestyle='--', label=f'median = {paired["lift"].median():.4f}')
    ax_c.set_xlabel("Pearson lift (pretrained - random)")
    ax_c.set_ylabel("Count (cell type × stage)")
    ax_c.set_title("C. Lift distribution")
    ax_c.legend()

    # ── Panel D: Top cell types by pretrained advantage ───────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ct_lift = paired.groupby(level=0)["lift"].mean().sort_values(ascending=True)
    top_n = 25
    ct_top = ct_lift.tail(top_n)
    bar_colors = [COLORS["pretrained"] if v >= 0 else COLORS["random"]
                  for v in ct_top.values]
    ax_d.barh(range(len(ct_top)), ct_top.values, color=bar_colors, alpha=0.7)
    ax_d.set_yticks(range(len(ct_top)))
    ax_d.set_yticklabels(ct_top.index, fontsize=6)
    ax_d.axvline(0, color='black', linewidth=0.8)
    ax_d.set_xlabel("Mean Pearson lift")
    ax_d.set_title(f"D. Top {top_n} cell types (pretrained advantage)")

    fig.suptitle("Figure 5: Developmental performance patterns",
                 fontsize=14, y=1.02)
    save_fig(fig, "fig5_developmental_patterns")
else:
    print("Skipping Fig 5: no per-track Pearson data")

# %% [markdown]
# ---
# ## Figure 6 — Conservation vs Expression Confounding
#
# Adapted from `04_analyze_attributions_celltypes_decima.ipynb`.
# Merge per-gene Pearson with zebrafish-human ortholog sequence identity.

# %%
if os.path.exists(ORTHO_SRC):
    print(f"Loading orthologs from: {ORTHO_SRC}")
    ortho = pd.read_csv(ORTHO_SRC)
    print(f"Ortholog table: {ortho.shape[0]} genes, columns: {list(ortho.columns)}")

    # Merge with gene-level Pearson (mean across replicates)
    gene_perf = gene_means_wide.reset_index()
    gene_perf = gene_perf.merge(ortho, left_on="gene", right_on="gene",
                                 how="inner")
    gene_perf["lift"] = gene_perf["pretrained"] - gene_perf["random"]
    gene_perf["perc_id"] = pd.to_numeric(gene_perf["source_perc_id"],
                                          errors="coerce")

    valid = gene_perf.dropna(subset=["perc_id", "pretrained", "random"]).copy()
    print(f"Genes with ortholog + performance: {len(valid)}")

    if len(valid) > 50:
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

        # ── Panel A: Pretrained Pearson vs % identity ─────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(valid.perc_id, valid.pretrained, s=3, alpha=0.3,
                    color=COLORS["pretrained"])
        rho, p = spearmanr(valid.perc_id, valid.pretrained)
        ax1.set_xlabel("Sequence identity to human (%)")
        ax1.set_ylabel("Pretrained Pearson r")
        ax1.set_title(f"A. Pretrained vs conservation\n(rho={rho:.3f}, p={p:.1e})")

        # ── Panel B: Random Pearson vs % identity ─────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(valid.perc_id, valid["random"], s=3, alpha=0.3,
                    color=COLORS["random"])
        rho, p = spearmanr(valid.perc_id, valid["random"])
        ax2.set_xlabel("Sequence identity to human (%)")
        ax2.set_ylabel("Random Pearson r")
        ax2.set_title(f"B. Random vs conservation\n(rho={rho:.3f}, p={p:.1e})")

        # ── Panel C: Lift vs % identity ───────────────────────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(valid.perc_id, valid.lift, s=3, alpha=0.3, color="grey")
        rho, p = spearmanr(valid.perc_id, valid.lift)
        ax3.axhline(0, color='black', linewidth=0.8)
        ax3.set_xlabel("Sequence identity to human (%)")
        ax3.set_ylabel("Lift (pretrained - random)")
        ax3.set_title(f"C. Lift vs conservation\n(rho={rho:.3f}, p={p:.1e})")

        # ── Panel D: Binned performance by conservation quartile ──────────────
        ax4 = fig.add_subplot(gs[1, 0])
        valid["id_bin"] = pd.qcut(valid.perc_id, q=4, duplicates="drop")
        bin_perf = valid.groupby("id_bin")[["pretrained", "random"]].mean()
        x = np.arange(len(bin_perf))
        width = 0.35
        ax4.bar(x - width/2, bin_perf.pretrained, width,
                color=COLORS["pretrained"], label="Pretrained", alpha=0.7)
        ax4.bar(x + width/2, bin_perf["random"], width,
                color=COLORS["random"], label="Random", alpha=0.7)
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"{iv.left:.0f}-{iv.right:.0f}%"
                             for iv in bin_perf.index], fontsize=8)
        ax4.set_xlabel("Conservation quartile (% identity)")
        ax4.set_ylabel("Mean Pearson r")
        ax4.set_title("D. Performance by conservation bin")
        ax4.legend(fontsize=8)

        # ── Panel E: Binned lift by conservation quartile ─────────────────────
        ax5 = fig.add_subplot(gs[1, 1])
        bin_lift = valid.groupby("id_bin")["lift"].agg(["mean", "sem"])
        ax5.bar(range(len(bin_lift)), bin_lift["mean"],
                yerr=bin_lift["sem"],
                color=COLORS["pretrained"], alpha=0.7, capsize=4)
        ax5.axhline(0, color='black', linewidth=0.8)
        ax5.set_xticks(range(len(bin_lift)))
        ax5.set_xticklabels([f"{iv.left:.0f}-{iv.right:.0f}%"
                             for iv in bin_lift.index], fontsize=8)
        ax5.set_xlabel("Conservation quartile (% identity)")
        ax5.set_ylabel("Mean lift")
        ax5.set_title("E. Pretrained advantage by conservation")

        # ── Panel F: Expression-matched analysis ──────────────────────────────
        ax6 = fig.add_subplot(gs[1, 2])
        # Match by expression level (mean_counts from any prediction)
        ref_ad = next(iter(ads.values()))
        expr_df = ref_ad.var.loc[ref_ad.var.dataset == "test",
                                  ["mean_counts"]].copy()
        expr_df.index.name = "gene"
        expr_df = expr_df.reset_index()
        valid_expr = valid.merge(expr_df, on="gene", how="left")
        valid_expr = valid_expr.dropna(subset=["mean_counts", "perc_id"])
        if len(valid_expr) > 50:
            valid_expr["expr_q"] = pd.qcut(
                valid_expr.mean_counts, q=3, labels=["Low", "Mid", "High"],
                duplicates="drop")
            for eq in ["Low", "Mid", "High"]:
                sub = valid_expr[valid_expr.expr_q == eq]
                if len(sub) < 10:
                    continue
                ax6.scatter(sub.perc_id, sub.lift, s=5, alpha=0.3, label=eq)
            ax6.axhline(0, color='black', linewidth=0.8)
            ax6.set_xlabel("Sequence identity to human (%)")
            ax6.set_ylabel("Lift (pretrained - random)")
            ax6.set_title("F. Lift vs conservation\n(expression-matched)")
            ax6.legend(fontsize=8, title="Expression")

        fig.suptitle("Figure 6: Conservation vs expression confounding",
                     fontsize=14, y=1.02)
        save_fig(fig, "fig6_conservation_analysis")
    else:
        print(f"Skipping Fig 6: only {len(valid)} genes with data")
else:
    print(f"Skipping Fig 6: ortholog file not found at {ORTHO_SRC}")

# %% [markdown]
# ---
# ## Figures 7 & 8 — Attribution by Genomic Region
#
# Adapted from `04_analyze_attributions_celltypes_decima.ipynb`.
# Loads regional attribution pkl files from `03_attributions/`.

# %%
# Load regional attribution data
region_dfs = []
for model_key, group_label in ATTR_MODELS.items():
    pkl_path = os.path.join(ATTR_DIR, f"{model_key}_regional_attributions.pkl")
    if not os.path.exists(pkl_path):
        print(f"WARNING: missing {pkl_path}")
        continue
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    df = raw['genes'].copy()
    df["model"] = model_key
    df["group"] = group_label
    region_dfs.append(df)
    print(f"Loaded {model_key}: {df.shape}")

if region_dfs:
    df_region = pd.concat(region_dfs, ignore_index=True)
    print(f"\nCombined regional data: {df_region.shape}")
else:
    df_region = pd.DataFrame()
    print("WARNING: No regional attribution data loaded")

# %%
# ── Figure 7: Attribution by genomic region (boxplot, log scale) ──────────────
# Include both base regions and distance-based regions (matching original)
BASE_REGION_COLS = ["Promoter", "Exons", "Introns", "Exon/Intron junctions"]
DIST_REGION_COLS = ["1k (all)", "1k-10k (all)", "10k-100k (all)", ">=100k (all)"]
ALL_REGION_COLS = BASE_REGION_COLS + [c for c in DIST_REGION_COLS
                                       if c in df_region.columns]
ALL_REGION_LABELS = {
    "Promoter": "Promoter", "Exons": "Exons", "Introns": "Introns",
    "Exon/Intron junctions": "Exon/Intron\njunctions",
    "1k (all)": "0–1 kb", "1k-10k (all)": "1–10 kb",
    "10k-100k (all)": "10–100 kb", ">=100k (all)": "≥100 kb",
}

if len(df_region) > 0:
    # Melt to long format
    df_long = df_region.melt(
        id_vars=["gene", "model", "group", "dataset"],
        value_vars=ALL_REGION_COLS,
        var_name="region", value_name="attribution",
    )
    # Filter to test genes and remove NaN/zero
    df_long = df_long[df_long.dataset == "test"].dropna(subset=["attribution"])
    df_long = df_long[df_long.attribution > 0]

    # Order regions by mean attribution (descending)
    region_order = (df_long.groupby("region").attribution.mean()
                    .sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df_long, x="region", y="attribution", hue="group",
                order=region_order, palette=COLORS, showfliers=False, ax=ax)
    ax.set_yscale('log')
    ax.set_xlabel("Genomic region")
    ax.set_ylabel("Attribution score (log scale)")
    ax.set_xticks(range(len(region_order)))
    ax.set_xticklabels([ALL_REGION_LABELS.get(r, r) for r in region_order],
                       rotation=30, ha='right')
    ax.set_title("Figure 7: Attribution by genomic region")

    # Mann-Whitney U tests per region
    for i, reg in enumerate(region_order):
        pre = df_long[(df_long.region == reg) &
                      (df_long.group == "pretrained")].attribution
        rnd = df_long[(df_long.region == reg) &
                      (df_long.group == "random")].attribution
        if len(pre) > 0 and len(rnd) > 0:
            _, p = mannwhitneyu(pre, rnd, alternative='two-sided')
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ymax = df_long[df_long.region == reg].attribution.quantile(0.95)
            ax.text(i, ymax * 1.5, stars, ha='center', fontsize=10,
                    fontweight='bold')

    ax.legend(title="Model type")
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    save_fig(fig, "fig7_attribution_by_region")

# %%
# ── Figure 8: CRE fold change by genomic region ──────────────────────────────
# Per-model median fold change approach (matching original notebook)
CRE_PAIRS = [
    ("Promoter CREs", "Promoter non-CREs", "Promoter"),
    ("Intronic CREs", "Intronic non-CREs", "Intronic"),
    ("1k (CREs)", "1k (non-CREs)", "0–1 kb"),
    ("1k-10k (CREs)", "1k-10k (non-CREs)", "1–10 kb"),
    ("10k-100k (CREs)", "10k-100k (non-CREs)", "10–100 kb"),
    (">=100k (CREs)", ">=100k (non-CREs)", "≥100 kb"),
]

if len(df_region) > 0:
    fc_rows = []
    for model_name in df_region["model"].unique():
        model_data = df_region[(df_region["model"] == model_name) &
                               (df_region["dataset"] == "test")]
        group = model_data["group"].iloc[0]
        for cre_col, noncre_col, label in CRE_PAIRS:
            if cre_col not in model_data.columns or noncre_col not in model_data.columns:
                continue
            cre_vals = model_data[cre_col].dropna()
            noncre_vals = model_data[noncre_col].dropna()
            if len(cre_vals) > 0 and len(noncre_vals) > 0:
                median_cre = cre_vals.median()
                median_noncre = noncre_vals.median()
                if median_noncre > 0:
                    fold_change = np.log2(median_cre / median_noncre)
                    fc_rows.append({
                        "region": label, "group": group,
                        "model": model_name, "log2_fc": fold_change,
                        "CRE_median": median_cre,
                        "non_CRE_median": median_noncre,
                    })
    df_fc = pd.DataFrame(fc_rows)

    if len(df_fc) > 0:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        region_order = [label for _, _, label in CRE_PAIRS
                        if label in df_fc.region.values]
        sns.boxplot(data=df_fc, x="region", y="log2_fc", hue="group",
                    order=region_order, palette=COLORS,
                    showfliers=False, width=0.6, linewidth=1.5, ax=ax)
        sns.stripplot(data=df_fc, x="region", y="log2_fc", hue="group",
                      order=region_order, palette=COLORS,
                      dodge=True, size=8, alpha=0.8,
                      edgecolor='white', linewidth=0.5, ax=ax)
        # Remove duplicate legend entries from stripplot
        handles, labels = ax.get_legend_handles_labels()
        n = len(COLORS)
        ax.legend(handles[:n], labels[:n], title="Model type")

        ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
        ax.set_xlabel("Genomic region", fontsize=12)
        ax.set_ylabel("log2(CRE / non-CRE attribution)", fontsize=12)
        ax.set_title("Figure 8: CRE enrichment of attribution signal", fontsize=14)
        ax.set_xticks(range(len(region_order)))
        ax.set_xticklabels(region_order, rotation=30, ha='right')
        ax.grid(True, alpha=0.2, axis='y')
        fig.tight_layout()
        save_fig(fig, "fig8_atac_fold_change")
    else:
        print("Skipping Fig 8: no valid CRE fold change data")

# %% [markdown]
# ---
# ## Figures 9 & 10 — Motif Clustering & Cell-Type Dendrogram
#
# Adapted from `designs_clustering_analysis.ipynb`.
# Parse MoDISco reports to build a motif × cell_type occurrence matrix.

# %%
def extract_modisco_patterns(h5_path, pattern_group="pos_patterns"):
    """Extract pattern CWMs and seqlet counts from a MoDISco report."""
    patterns = []
    with h5py.File(h5_path, 'r') as f:
        if pattern_group not in f:
            return patterns
        g = f[pattern_group]
        for pname in sorted(g.keys(), key=lambda x: int(x.split('_')[1])):
            pat = g[pname]
            cwm = pat['contrib_scores'][:]  # (L, 4)
            n_seqlets = int(pat['seqlets']['n_seqlets'][()].item())
            # Compute information content as max |CWM| per position
            ic = np.abs(cwm).sum()
            patterns.append({
                "pattern": f"{pattern_group}/{pname}",
                "n_seqlets": n_seqlets,
                "ic_total": ic,
                "cwm": cwm,
            })
    return patterns


def build_motif_matrix(modisco_dir, cell_types, min_seqlets=20):
    """Build motif occurrence matrix (motifs × cell types).

    For each cell type, count the number of seqlets per pattern.
    We align patterns across cell types by their CWM similarity.
    As a simpler approach: just count total pos patterns and their seqlet counts.
    """
    # First pass: collect all patterns per cell type
    ct_patterns = {}
    for ct in cell_types:
        h5_path = os.path.join(modisco_dir, ct, "modisco_report.h5")
        if not os.path.exists(h5_path):
            print(f"WARNING: missing {h5_path}")
            continue
        pats = extract_modisco_patterns(h5_path, "pos_patterns")
        # Filter by min seqlets
        pats = [p for p in pats if p["n_seqlets"] >= min_seqlets]
        ct_patterns[ct] = pats
        print(f"  {ct}: {len(pats)} positive patterns (>={min_seqlets} seqlets)")

    if not ct_patterns:
        return pd.DataFrame()

    # Build a simple matrix: rows = pattern index, cols = cell type
    # Use top-N patterns per cell type (by seqlet count)
    max_patterns = max(len(v) for v in ct_patterns.values())

    # Matrix: pattern_rank × cell_type (seqlet count, 0 if fewer patterns)
    matrix = pd.DataFrame(0, index=[f"pattern_{i}" for i in range(max_patterns)],
                          columns=list(ct_patterns.keys()))
    for ct, pats in ct_patterns.items():
        for i, pat in enumerate(sorted(pats, key=lambda x: -x["n_seqlets"])):
            matrix.loc[f"pattern_{i}", ct] = pat["n_seqlets"]

    # Remove all-zero rows
    matrix = matrix.loc[matrix.sum(axis=1) > 0]
    return matrix


print("Building motif occurrence matrix from MoDISco reports...")
motif_matrix = build_motif_matrix(MODISC_DIR, FOCAL_CELLTYPES, min_seqlets=20)
print(f"Motif matrix shape: {motif_matrix.shape}")

# %%
# ── Figure 9: Motif clustering heatmap ────────────────────────────────────────
if motif_matrix.shape[0] > 2 and motif_matrix.shape[1] > 1:
    # Z-score normalize rows (patterns)
    expr_z = motif_matrix.sub(motif_matrix.mean(axis=1), axis=0) \
                          .div(motif_matrix.std(axis=1).replace(0, 1), axis=0)

    # Row clustering (patterns)
    dist_rows = pdist(expr_z.values, metric='euclidean')
    linkage_rows = linkage(dist_rows, method='average')

    # Column clustering (cell types)
    dist_cols = pdist(expr_z.T.values, metric='correlation')
    linkage_cols = linkage(dist_cols, method='average')

    fig = plt.figure(figsize=(12, 12))

    # Row dendrogram (left)
    ax_dr = fig.add_axes([0.02, 0.05, 0.12, 0.65])
    dendro_r = dendrogram(linkage_rows, orientation='left',
                          labels=expr_z.index.tolist(),
                          leaf_font_size=7, ax=ax_dr, no_labels=True)
    ax_dr.set_xlabel('Distance')

    # Column dendrogram (top) — with extra room for labels
    ax_dc = fig.add_axes([0.18, 0.75, 0.70, 0.12])
    dendro_c = dendrogram(linkage_cols, labels=expr_z.columns.tolist(),
                          leaf_rotation=90, leaf_font_size=0, ax=ax_dc)
    ax_dc.set_ylabel('1-corr')
    ax_dc.tick_params(axis='x', labelbottom=False)

    # Reorder matrix
    ordered = expr_z.iloc[dendro_r['leaves'], :]
    ordered = ordered.iloc[:, dendro_c['leaves']]

    # Heatmap
    ax_h = fig.add_axes([0.18, 0.05, 0.70, 0.65])
    im = ax_h.imshow(ordered.values, aspect='auto', cmap='RdBu_r',
                     vmin=-2, vmax=2)
    ax_h.set_yticks(range(ordered.shape[0]))
    ax_h.set_yticklabels(ordered.index, fontsize=6)
    ax_h.set_xticks(range(ordered.shape[1]))
    ax_h.set_xticklabels(
        [c.replace('_', ' ') for c in ordered.columns],
        rotation=45, ha='right', fontsize=10)

    # Colorbar
    cbar_ax = fig.add_axes([0.90, 0.05, 0.02, 0.65])
    plt.colorbar(im, cax=cbar_ax, label='Z-score (seqlet count)')

    fig.suptitle("Figure 9: Motif clustering across cell types",
                 fontsize=14, x=0.55, y=0.95)
    save_fig(fig, "fig9_motif_clustering")
else:
    print("Skipping Fig 9: insufficient motif data")

# %%
# ── Figure 10: Cell-type dendrogram ───────────────────────────────────────────
if motif_matrix.shape[1] > 2:
    # Z-score normalize
    expr_z = motif_matrix.sub(motif_matrix.mean(axis=1), axis=0) \
                          .div(motif_matrix.std(axis=1).replace(0, 1), axis=0)

    d = pdist(expr_z.T.values, metric='correlation')
    Z = linkage(d, method='average')

    fig, ax = plt.subplots(figsize=(14, 5))
    dendro = dendrogram(
        Z,
        labels=[c.replace('_', ' ') for c in expr_z.columns],
        leaf_rotation=45,
        leaf_font_size=11,
        color_threshold=0.85 * Z[:, 2].max(),
        ax=ax,
    )
    ax.set_ylabel('1 − correlation')
    ax.set_title('Figure 10: Cell-type similarity based on motif profiles')
    fig.tight_layout()
    save_fig(fig, "fig10_celltype_dendrogram")
else:
    print("Skipping Fig 10: insufficient cell types")

# %% [markdown]
# ---
# ## Summary

# %%
print("\n" + "=" * 70)
print("MANUSCRIPT FIGURE GENERATION COMPLETE")
print("=" * 70)
print(f"\nAll figures saved to: {FIG_DIR}")
for f in sorted(os.listdir(FIG_DIR)):
    fpath = os.path.join(FIG_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {f:45s}  ({size_kb:.0f} KB)")
