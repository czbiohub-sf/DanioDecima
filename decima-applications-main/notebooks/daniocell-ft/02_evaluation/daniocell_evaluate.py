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
# # DanioCell Decima — Post-Training Evaluation (Phase 2)
#
# Evaluates 8 DanioCell Decima models (4 pretrained, 4 random).
#
# **Phase 2a:** Per-gene and per-track Pearson correlation analysis
# **Phase 2b:** Marker gene analysis (z-score AUROC/AUPRC via decima.evaluate)
# **Phase 2c:** DanioCell-specific stage-group analysis (14 developmental stages)
#
# Run with: `module load anaconda && source activate gReLu`

# %%
import numpy as np
import pandas as pd
import anndata
import os
import sys
import warnings
from glob import glob
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning)

# Import decima evaluation functions
src_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..',
    'decima-main', 'src', 'decima'
))
sys.path.insert(0, src_dir)
from evaluate import marker_zscores, compare_marker_zscores, compute_marker_metrics

# %% [markdown]
# ## Configuration

# %%
PRED_DIR = "/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/01_predictions"
OUT_DIR = "/hpc/scratch/group.data.science/yang-joon.kim/daniodecima-daniocell/02_evaluation"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Experiment names matching output filenames from Phase 1
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

CELLTYPE_KEY = "identity.super"
STAGE_KEY = "stage.group"

# Plot style
sns.set_style("whitegrid")
COLORS = {"pretrained": "#2166ac", "random": "#b2182b"}

# %% [markdown]
# ## Load predictions

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

# Quick sanity check: test-gene Pearson matches training report
for name, ad in ads.items():
    test_r = ad.var.loc[ad.var.dataset == 'test', 'pearson'].mean()
    group = "pretrained" if "pretrained" in name else "random"
    print(f"  {name}: test Pearson = {test_r:.4f}  ({group})")

# %% [markdown]
# ## Phase 2a: Correlation Analysis

# %% [markdown]
# ### Build summary DataFrame

# %%
def build_pearson_summary(ads):
    """Collect per-gene and per-track Pearson into a tidy DataFrame."""
    rows = []
    for name, ad in ads.items():
        group = "pretrained" if "pretrained" in name else "random"
        # Per-gene (test set only)
        test_mask = ad.var.dataset == "test"
        gene_pearsons = ad.var.loc[test_mask, "pearson"].values
        for r in gene_pearsons:
            rows.append({"exp": name, "group": group, "metric": "gene_pearson",
                         "value": r})
        # Per-track (test genes)
        if "test_pearson" in ad.obs.columns:
            track_pearsons = ad.obs["test_pearson"].values
            for r in track_pearsons:
                rows.append({"exp": name, "group": group, "metric": "track_pearson",
                             "value": r})
    return pd.DataFrame(rows)

df_pearson = build_pearson_summary(ads)
print(f"Summary rows: {len(df_pearson)}")

# %% [markdown]
# ### Plot 1: Per-gene Pearson histogram (test set) — pretrained vs random

# %%
fig, ax = plt.subplots(figsize=(7, 3))
for group, color in COLORS.items():
    vals = df_pearson[(df_pearson.group == group) & (df_pearson.metric == "gene_pearson")]
    ax.hist(vals.value, bins=60, alpha=0.55, color=color, label=group, density=True)
ax.set_xlabel("Pearson correlation per gene (test set)")
ax.set_ylabel("Density")
ax.legend()
ax.set_title("Per-gene Pearson: pretrained vs random")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_gene_pearson_histogram.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR, "01_gene_pearson_histogram.png"), dpi=150)
plt.close(fig)
print("Saved: 01_gene_pearson_histogram")

# %% [markdown]
# ### Plot 2: Per-track Pearson histogram (test genes)

# %%
fig, ax = plt.subplots(figsize=(7, 3))
for group, color in COLORS.items():
    vals = df_pearson[(df_pearson.group == group) & (df_pearson.metric == "track_pearson")]
    if len(vals) == 0:
        continue
    ax.hist(vals.value, bins=60, alpha=0.55, color=color, label=group, density=True)
ax.set_xlabel("Pearson correlation per track (test genes)")
ax.set_ylabel("Density")
ax.legend()
ax.set_title("Per-track Pearson: pretrained vs random")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_track_pearson_histogram.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR, "02_track_pearson_histogram.png"), dpi=150)
plt.close(fig)
print("Saved: 02_track_pearson_histogram")

# %% [markdown]
# ### Plot 3: Pretrained vs random scatter (per-gene Pearson, mean across reps)

# %%
# Average Pearson across replicates for each gene
def mean_pearson_per_gene(ads, names, split="test"):
    """Compute mean Pearson per gene across a set of replicates."""
    gene_pearsons = []
    for name in names:
        if name not in ads:
            continue
        ad = ads[name]
        mask = ad.var.dataset == split
        gene_pearsons.append(ad.var.loc[mask, "pearson"].values)
    return np.nanmean(np.stack(gene_pearsons), axis=0)

ref_ad = next(iter(ads.values()))
test_genes = ref_ad.var_names[ref_ad.var.dataset == "test"]

pretrained_mean = mean_pearson_per_gene(ads, PRETRAINED_NAMES)
random_mean = mean_pearson_per_gene(ads, RANDOM_NAMES)

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(random_mean, pretrained_mean, s=3, alpha=0.3, color="grey")
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, '--', color='black', linewidth=0.8, alpha=0.5)
ax.set_xlabel("Random init (mean Pearson)")
ax.set_ylabel("Pretrained (mean Pearson)")
ax.set_title("Per-gene Pearson: pretrained vs random (test set)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_pretrained_vs_random_scatter.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR, "03_pretrained_vs_random_scatter.png"), dpi=150)
plt.close(fig)
print("Saved: 03_pretrained_vs_random_scatter")

# %% [markdown]
# ### Plot 4: Summary bar chart — mean Pearson +/- std by group

# %%
summary_rows = []
for name, ad in ads.items():
    group = "pretrained" if "pretrained" in name else "random"
    test_mask = ad.var.dataset == "test"
    mean_r = ad.var.loc[test_mask, "pearson"].mean()
    summary_rows.append({"exp": name, "group": group, "mean_pearson": mean_r})
df_summary = pd.DataFrame(summary_rows)

fig, ax = plt.subplots(figsize=(4, 4))
for i, (group, color) in enumerate(COLORS.items()):
    sub = df_summary[df_summary.group == group]
    mean_val = sub.mean_pearson.mean()
    std_val = sub.mean_pearson.std()
    ax.bar(i, mean_val, yerr=std_val, color=color, alpha=0.7, capsize=5, width=0.6)
    # Individual points
    ax.scatter([i] * len(sub), sub.mean_pearson, color='black', s=30, zorder=3)
ax.set_xticks([0, 1])
ax.set_xticklabels(["pretrained", "random"])
ax.set_ylabel("Mean Pearson (test genes)")
ax.set_title("Model comparison")

# Mann-Whitney U test
pre_vals = df_summary[df_summary.group == "pretrained"].mean_pearson.values
rnd_vals = df_summary[df_summary.group == "random"].mean_pearson.values
if len(pre_vals) > 1 and len(rnd_vals) > 1:
    stat, pval = mannwhitneyu(pre_vals, rnd_vals, alternative='greater')
    ax.text(0.5, 0.95, f"MWU p={pval:.3f}", transform=ax.transAxes,
            ha='center', va='top', fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_summary_bar_chart.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR, "04_summary_bar_chart.png"), dpi=150)
plt.close(fig)
print("Saved: 04_summary_bar_chart")

# %% [markdown]
# ### Plot 5: Per-replicate consistency — violin plot of pretrained Pearson distributions

# %%
df_pre = df_pearson[(df_pearson.group == "pretrained") &
                    (df_pearson.metric == "gene_pearson")].copy()
# Extract replicate number from exp name
df_pre["rep"] = df_pre.exp.str.extract(r'rep(\d)').astype(int)

fig, ax = plt.subplots(figsize=(6, 3.5))
sns.violinplot(data=df_pre, x="rep", y="value", inner="quartile",
               color=COLORS["pretrained"], alpha=0.7, ax=ax)
ax.set_xlabel("Pretrained replicate")
ax.set_ylabel("Pearson correlation (test genes)")
ax.set_title("Cross-replicate consistency")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "05_replicate_violin.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR, "05_replicate_violin.png"), dpi=150)
plt.close(fig)
print("Saved: 05_replicate_violin")

# %% [markdown]
# ## Phase 2b: Marker Gene Analysis
#
# Uses decima.evaluate functions: marker_zscores, compare_marker_zscores, compute_marker_metrics

# %%
# Use the best pretrained replicate for marker analysis
best_pretrained = max(
    [(n, ads[n].var.loc[ads[n].var.dataset == 'test', 'pearson'].mean())
     for n in PRETRAINED_NAMES if n in ads],
    key=lambda x: x[1]
)[0]
print(f"Best pretrained model: {best_pretrained}")

ad_best = ads[best_pretrained]

# Filter to test genes for marker analysis
ad_test = ad_best[:, ad_best.var.dataset == "test"].copy()
print(f"Test AnnData shape: {ad_test.shape}")

# %% [markdown]
# ### Compute marker z-scores and metrics

# %%
print("Computing marker z-scores (observed vs predicted)...")
marker_df = compare_marker_zscores(ad_test, key=CELLTYPE_KEY)
print(f"Marker DataFrame shape: {marker_df.shape}")

print("Computing marker metrics (AUROC, AUPRC) per cell type...")
metrics_df = compute_marker_metrics(marker_df, key=CELLTYPE_KEY, tp_cutoff=1)
metrics_df = metrics_df.sort_values("auroc", ascending=False)
print(f"Cell types with valid AUROC: {metrics_df.auroc.notna().sum()}")
print(f"Mean AUROC: {metrics_df.auroc.mean():.3f}")
print(f"Mean AUPRC: {metrics_df.auprc.mean():.3f}")

# Save metrics table
metrics_df.to_csv(os.path.join(OUT_DIR, "marker_metrics_per_celltype.csv"), index=False)

# %% [markdown]
# ### Plot 6: Per-cell-type AUROC bar chart (top 30)

# %%
top_ct = metrics_df.dropna(subset=["auroc"]).head(30)

fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(range(len(top_ct)), top_ct.auroc.values, color=COLORS["pretrained"], alpha=0.7)
ax.set_yticks(range(len(top_ct)))
ax.set_yticklabels(top_ct[CELLTYPE_KEY].values, fontsize=7)
ax.set_xlabel("AUROC")
ax.set_title(f"Marker gene detection — top 30 cell types\n({best_pretrained})")
ax.invert_yaxis()
ax.axvline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "06_celltype_auroc_bar.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR, "06_celltype_auroc_bar.png"), dpi=150)
plt.close(fig)
print("Saved: 06_celltype_auroc_bar")

# %% [markdown]
# ### Plot 7: Per-cell-type AUPRC bar chart (top 30)

# %%
top_ct_auprc = metrics_df.dropna(subset=["auprc"]).sort_values("auprc", ascending=False).head(30)

fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(range(len(top_ct_auprc)), top_ct_auprc.auprc.values,
        color=COLORS["pretrained"], alpha=0.7)
ax.set_yticks(range(len(top_ct_auprc)))
ax.set_yticklabels(top_ct_auprc[CELLTYPE_KEY].values, fontsize=7)
ax.set_xlabel("AUPRC")
ax.set_title(f"Marker gene precision-recall — top 30 cell types\n({best_pretrained})")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "07_celltype_auprc_bar.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR, "07_celltype_auprc_bar.png"), dpi=150)
plt.close(fig)
print("Saved: 07_celltype_auprc_bar")

# %% [markdown]
# ### Plot 8: Predicted vs observed z-score scatter, colored by AUROC

# %%
# Merge AUROC into marker_df
marker_with_auroc = marker_df.merge(
    metrics_df[[CELLTYPE_KEY, "auroc"]], on=CELLTYPE_KEY, how="left"
)

# Subsample for plotting (too many points otherwise)
np.random.seed(42)
n_sample = min(50000, len(marker_with_auroc))
plot_df = marker_with_auroc.sample(n=n_sample, random_state=42)

fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(plot_df.score_obs, plot_df.score_pred, c=plot_df.auroc,
                cmap="RdYlBu_r", s=1, alpha=0.3, vmin=0.5, vmax=1.0)
plt.colorbar(sc, ax=ax, label="AUROC")
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, '--', color='black', linewidth=0.8, alpha=0.5)
ax.set_xlabel("Observed z-score")
ax.set_ylabel("Predicted z-score")
ax.set_title("Marker z-scores: observed vs predicted")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "08_zscore_scatter.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR, "08_zscore_scatter.png"), dpi=150)
plt.close(fig)
print("Saved: 08_zscore_scatter")

# %% [markdown]
# ### Plot 9: Marker z-score heatmap (top cell types × top marker genes)

# %%
# Select top 20 cell types by AUROC
top20_ct = metrics_df.dropna(subset=["auroc"]).head(20)[CELLTYPE_KEY].values

# For each top cell type, find the top 3 marker genes (by observed z-score)
top_markers = []
for ct in top20_ct:
    ct_markers = marker_df[marker_df[CELLTYPE_KEY] == ct].nlargest(3, "score_obs")
    top_markers.extend(ct_markers.gene.values)
top_markers = list(dict.fromkeys(top_markers))  # unique, preserve order

# Build heatmap matrix (predicted z-scores)
heatmap_data = marker_df[
    (marker_df[CELLTYPE_KEY].isin(top20_ct)) & (marker_df.gene.isin(top_markers))
].pivot_table(index=CELLTYPE_KEY, columns="gene", values="score_pred")
heatmap_data = heatmap_data.loc[top20_ct, top_markers]

fig, ax = plt.subplots(figsize=(max(10, len(top_markers) * 0.3), 6))
sns.heatmap(heatmap_data, cmap="RdBu_r", center=0, ax=ax,
            xticklabels=True, yticklabels=True, linewidths=0.2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
ax.set_title("Predicted marker z-scores (top 20 cell types)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "09_marker_zscore_heatmap.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR, "09_marker_zscore_heatmap.png"), dpi=150)
plt.close(fig)
print("Saved: 09_marker_zscore_heatmap")

# %% [markdown]
# ## Phase 2c: DanioCell-Specific Stage Analysis
#
# DanioCell spans 14 developmental stage groups (3-120 hpf).
# Assess whether prediction quality varies across development.

# %% [markdown]
# ### Plot 10: Per-stage-group mean Pearson

# %%
# Compute mean test-gene Pearson per track, then average by stage group
stage_pearson_rows = []
for name, ad in ads.items():
    group = "pretrained" if "pretrained" in name else "random"
    if "test_pearson" not in ad.obs.columns:
        continue
    for stage in ad.obs[STAGE_KEY].unique():
        mask = ad.obs[STAGE_KEY] == stage
        mean_r = ad.obs.loc[mask, "test_pearson"].mean()
        stage_pearson_rows.append({
            "exp": name, "group": group, "stage": stage.strip(), "mean_pearson": mean_r
        })
df_stage = pd.DataFrame(stage_pearson_rows)

# Sort stages numerically
def stage_sort_key(s):
    """Extract first number from stage string for sorting."""
    return int(s.split('-')[0].strip())

stage_order = sorted(df_stage.stage.unique(), key=stage_sort_key)

fig, ax = plt.subplots(figsize=(10, 4))
for group, color in COLORS.items():
    sub = df_stage[df_stage.group == group]
    stage_means = sub.groupby("stage").mean_pearson.agg(["mean", "std"])
    stage_means = stage_means.loc[[s for s in stage_order if s in stage_means.index]]
    ax.errorbar(range(len(stage_means)), stage_means["mean"],
                yerr=stage_means["std"], marker='o', color=color,
                label=group, capsize=3, linewidth=1.5)
ax.set_xticks(range(len(stage_order)))
ax.set_xticklabels([f"{s} hpf" for s in stage_order], rotation=45, ha='right', fontsize=8)
ax.set_xlabel("Developmental stage group")
ax.set_ylabel("Mean Pearson (test genes)")
ax.set_title("Prediction accuracy across development")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "10_stage_pearson.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR, "10_stage_pearson.png"), dpi=150)
plt.close(fig)
print("Saved: 10_stage_pearson")

# %% [markdown]
# ### Plot 11: Heatmap — cell type × stage group Pearson per track (best pretrained model)

# %%
# Build per-track Pearson for test genes, pivoted by cell type and stage
ad_bp = ads[best_pretrained]
if "test_pearson" in ad_bp.obs.columns:
    track_df = ad_bp.obs[[CELLTYPE_KEY, STAGE_KEY, "test_pearson"]].copy()
    track_df[STAGE_KEY] = track_df[STAGE_KEY].str.strip()

    # Average across tracks sharing same (cell type, stage)
    pivot = track_df.pivot_table(
        index=CELLTYPE_KEY, columns=STAGE_KEY, values="test_pearson", aggfunc="mean"
    )
    # Sort columns by stage, rows by mean Pearson
    pivot = pivot[[s for s in stage_order if s in pivot.columns]]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    # Show top 40 cell types for readability
    pivot_top = pivot.head(40)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(pivot_top, cmap="RdYlBu_r", vmin=0, vmax=1, ax=ax,
                xticklabels=True, yticklabels=True, linewidths=0.2)
    ax.set_xticklabels([f"{s} hpf" for s in pivot_top.columns],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    ax.set_title(f"Per-track Pearson (test genes) — top 40 cell types\n({best_pretrained})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "11_celltype_stage_heatmap.pdf"), dpi=150)
    fig.savefig(os.path.join(FIG_DIR, "11_celltype_stage_heatmap.png"), dpi=150)
    plt.close(fig)
    print("Saved: 11_celltype_stage_heatmap")
else:
    print("Skipping plot 11: test_pearson not found in obs")

# %% [markdown]
# ## Summary statistics

# %%
print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)

for group in ["pretrained", "random"]:
    names = PRETRAINED_NAMES if group == "pretrained" else RANDOM_NAMES
    test_rs = [ads[n].var.loc[ads[n].var.dataset == 'test', 'pearson'].mean()
               for n in names if n in ads]
    print(f"\n{group.upper()} ({len(test_rs)} models):")
    print(f"  Test gene Pearson:  {np.mean(test_rs):.4f} +/- {np.std(test_rs):.4f}")
    if "test_pearson" in next(iter(ads.values())).obs.columns:
        track_rs = [ads[n].obs['test_pearson'].mean() for n in names if n in ads]
        print(f"  Test track Pearson: {np.mean(track_rs):.4f} +/- {np.std(track_rs):.4f}")

print(f"\nMarker analysis (best model: {best_pretrained}):")
print(f"  Mean AUROC:  {metrics_df.auroc.mean():.3f}")
print(f"  Mean AUPRC:  {metrics_df.auprc.mean():.3f}")
print(f"  Cell types with AUROC > 0.8: {(metrics_df.auroc > 0.8).sum()}")

# Save full summary
metrics_df.to_csv(os.path.join(OUT_DIR, "marker_metrics_per_celltype.csv"), index=False)
df_summary.to_csv(os.path.join(OUT_DIR, "model_summary.csv"), index=False)
print(f"\nResults saved to: {OUT_DIR}")
print(f"Figures saved to: {FIG_DIR}")
