#!/usr/bin/env python3
"""Generate DanioCell fine-tuning results slide deck."""

import sys
from pathlib import Path

# Add pptx_helpers to path
_helpers_dirs = [
    *Path.home().glob(".claude/plugins/marketplaces/*/plugins/generate-slides/scripts"),
    *Path.home().glob(".claude/plugins/cache/*/generate-slides/*/scripts"),
]
for d in _helpers_dirs:
    if (d / "pptx_helpers.py").exists():
        sys.path.insert(0, str(d))
        break

from pptx_helpers import *

# Paths
FIG_DIR = Path("/hpc/scratch/group.data.science/yang-joon.kim/"
               "daniodecima-daniocell/02_evaluation/manuscript_figures")
OUT = Path("/hpc/projects/data.science/yangjoon.kim/step/docs/slides/"
           "daniocell-finetuning-2026-04-08.pptx")


def slide_01_title(prs):
    slide = add_blank_slide(prs)
    make_title_slide(
        slide,
        "DanioDecima x DanioCell:\nScaling sequence-to-expression fine-tuning\nto a richer atlas",
        "Yang-Joon Kim, Mathias Voges",
        "CZ Biohub San Francisco",
        "April 2026",
    )


def slide_02_takeaway(prs):
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "We validated a fine-tuning framework on Zebrahub — can it scale to richer data?")
    # no accent line per user preference
    add_bullets(slide, LEFT_MARGIN, Inches(1.4), CONTENT_W, Inches(2.8), [
        ("Framework validated: ",
         "Same pipeline, same biology — pretrained >> random, no memorization"),
        ("Better data in: ",
         "DanioCell has 3.4x more pseudobulks and 2.5x better cell-type x stage coverage"),
        ("Model ready: ",
         "Gene-level Pearson identical to DanioDecima (0.38); attribution confirms biological signal"),
    ])
    add_callout_box(slide, LEFT_MARGIN, Inches(4.8), CONTENT_W, Inches(0.8),
                    "Bottom line: ",
                    "DanioCell model is ready for in silico mutagenesis and regulatory design")


def slide_03_dataset(prs):
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "DanioCell provides 3.4x more pseudobulks and 2.5x denser developmental coverage")
    # no accent line per user preference

    data = [
        ["Property", "Zebrahub", "DanioCell", "Change"],
        ["Pseudobulk tracks", "304", "1,047", "3.4x"],
        ["Cell types", "154", "153", "comparable"],
        ["Developmental stages", "10 (10hpf-10dpf)", "14 (3-120 hpf)", "1.4x"],
        ["Cell-type x stage coverage", "19.7%", "48.9%", "2.5x"],
        ["Total genes", "31,767", "28,201", "comparable"],
        ["Test genes", "7,289", "6,405", "comparable"],
    ]
    add_table(slide, LEFT_MARGIN, Inches(1.4), CONTENT_W, Inches(3.5),
              rows=7, cols=4, data=data,
              col_widths=[Inches(3.8), Inches(2.8), Inches(2.8), Inches(2.3)],
              highlight_cells={
                  (1, 3): GREEN, (3, 3): GREEN, (4, 3): GREEN,
              })
    add_callout_box(slide, LEFT_MARGIN, Inches(5.4), CONTENT_W, Inches(0.7),
                    "Key advantage: ",
                    "DanioCell's denser coverage means the model sees more developmental diversity during training",
                    font_size=18)


def slide_04_setup(prs):
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "Same architecture and loss — only the atlas and init strategy differ")
    # no accent line per user preference

    # Left column: DanioDecima
    add_textbox(slide, LEFT_MARGIN, Inches(1.4), Inches(5.5), Inches(0.4),
                "DanioDecima (Zebrahub) — Mathias", font_size=20, color=BLUE)
    add_bullets(slide, LEFT_MARGIN, Inches(1.9), Inches(5.5), Inches(2.5), [
        "16 experiments: 4 init x 4 replicates",
        "Random, Mouse-Borzoi, Human-Borzoi, Human-Decima",
        "10 timepoints, 304 pseudobulks",
        "TensorBoard logging, Ray Tune sweep",
    ], font_size=18)

    # Right column: DanioCell
    add_textbox(slide, Inches(7.0), Inches(1.4), Inches(5.5), Inches(0.4),
                "DanioCell Fine-Tuning — Yang-Joon", font_size=20, color=BLUE)
    add_bullets(slide, Inches(7.0), Inches(1.9), Inches(5.5), Inches(2.5), [
        "8 experiments: 2 init x 4 replicates",
        "Random (lr=3e-6), Pretrained Human-Decima (lr=3e-5)",
        "14 stage groups, 1,047 pseudobulks",
        "CSVLogger, deterministic training (CUDA)",
    ], font_size=18)

    add_callout_box(slide, LEFT_MARGIN, Inches(5.0), CONTENT_W, Inches(0.7),
                    "Rationale: ",
                    "Used best init strategy (Human-Decima) identified by Mathias's sweep",
                    font_size=18)


def slide_05_training(prs):
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "Pretrained models converge in 2-3 epochs; random oscillates through epoch 30")
    # no accent line per user preference
    add_figure(slide, FIG_DIR / "fig1_training_curves.png",
               LEFT_MARGIN, Inches(1.3), width=Inches(10.5))
    add_bullets(slide, LEFT_MARGIN, Inches(5.6), CONTENT_W, Inches(1.2), [
        ("Pretrained: ", "val_pearson ~0.76, stable from epoch 3"),
        ("Random: ", "val_pearson ~0.50, high variance through epoch 30"),
    ], font_size=18)


def slide_06_headtohead(prs):
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "Gene-level Pearson is identical (0.38); pretrained lift is actually larger (+0.23 vs +0.19)")
    # no accent line per user preference

    data = [
        ["Metric", "Zebrahub", "DanioCell", ""],
        ["Track Pearson (pretrained)", "0.81", "0.76", "More rare cell types"],
        ["Track Pearson (random)", "0.62", "0.53", ""],
        ["Track lift", "+0.19", "+0.23", "DanioCell > Zebrahub"],
        ["Gene Pearson (pretrained)", "0.38", "0.38", "Identical"],
        ["Gene Pearson (random)", "0.24", "0.23", ""],
        ["Gene lift", "+0.14", "+0.15", "Identical"],
    ]
    add_table(slide, LEFT_MARGIN, Inches(1.4), CONTENT_W, Inches(3.2),
              rows=7, cols=4, data=data,
              col_widths=[Inches(3.5), Inches(2.2), Inches(2.2), Inches(3.8)],
              highlight_cells={
                  (3, 2): GREEN, (3, 3): GREEN,
                  (4, 2): GREEN, (4, 3): GREEN,
                  (6, 2): GREEN, (6, 3): GREEN,
              })
    add_callout_box(slide, LEFT_MARGIN, Inches(5.2), CONTENT_W, Inches(0.8),
                    "Interpretation: ",
                    "Lower absolute track Pearson reflects more cell types (incl. rare), "
                    "but the lift from pretraining is larger — the framework works even better with richer data",
                    font_size=18)


def slide_07_comparison(prs):
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "80% of test genes are better predicted by the pretrained model (MWU p=0.014)")
    # no accent line per user preference
    add_figure(slide, FIG_DIR / "fig3_model_comparison.png",
               LEFT_MARGIN, Inches(1.2), height=Inches(5.8))


def slide_08_developmental(prs):
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "Pretraining helps most where prediction is hardest: late development and specialized cell types")
    # no accent line per user preference
    add_figure(slide, FIG_DIR / "fig5_developmental_patterns.png",
               LEFT_MARGIN, Inches(1.2), height=Inches(5.8))


def slide_09_conservation(prs):
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "Pretrained lift is independent of gene conservation (rho=-0.07) — not memorizing")
    # no accent line per user preference
    add_figure(slide, FIG_DIR / "fig6_conservation_analysis.png",
               LEFT_MARGIN, Inches(1.3), height=Inches(5.0))
    add_textbox(slide, LEFT_MARGIN, Inches(6.6), CONTENT_W, Inches(0.5),
                "Replicates DanioDecima finding: model learned regulatory grammar, not conserved sequences",
                font_size=16, color=GREY)


def slide_10_attribution(prs):
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "Pretrained model attributes higher importance to regulatory regions (all p < 0.001)")
    # no accent line per user preference
    add_figure(slide, FIG_DIR / "fig7_attribution_by_region.png",
               LEFT_MARGIN, Inches(1.3), width=Inches(10.5))
    add_bullets(slide, LEFT_MARGIN, Inches(5.8), CONTENT_W, Inches(1.2), [
        ("Promoter >> distal: ", "expected gradient of regulatory importance"),
        ("Pretrained >> random: ", "at all 8 regions — learned regulatory grammar"),
    ], font_size=18)


def slide_11_cre(prs):
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "Pretrained model preferentially attributes importance to ATAC-accessible CRE regions")
    # no accent line per user preference
    add_figure(slide, FIG_DIR / "fig8_atac_fold_change.png",
               LEFT_MARGIN, Inches(1.3), height=Inches(4.2))
    add_bullets(slide, LEFT_MARGIN, Inches(5.8), CONTENT_W, Inches(1.2), [
        ("Promoter CREs: ", "log2 FC ~0.85 — strong enrichment in accessible chromatin"),
        ("Random model: ", "near-zero enrichment at distal regions"),
    ], font_size=18)


def slide_12_motifs(prs):
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "MoDISco recovers lineage-specific motif architectures across 10 focal cell types")
    # no accent line per user preference

    # Side by side: heatmap left, dendrogram right
    add_figure(slide, FIG_DIR / "fig9_motif_clustering.png",
               LEFT_MARGIN, Inches(1.3), height=Inches(4.5))
    add_figure(slide, FIG_DIR / "fig10_celltype_dendrogram.png",
               Inches(6.8), Inches(2.5), width=Inches(5.5))
    add_textbox(slide, LEFT_MARGIN, Inches(6.2), CONTENT_W, Inches(0.5),
                "Lineage clustering: neurons-notochord, epidermis-radial glia, cardiac muscle-somite",
                font_size=16, color=GREY)


def slide_13_summary(prs):
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "All DanioDecima findings replicate with DanioCell — the framework generalizes")
    # no accent line per user preference

    data = [
        ["Evaluation", "DanioDecima (Zebrahub)", "DanioCell", "Status"],
        ["Gene Pearson (pretrained)", "0.38", "0.38", "Matched"],
        ["Pretrained >> random", "Yes (p < 0.001)", "Yes (p = 0.014)", "Replicated"],
        ["Conservation independence", "rho ~ 0", "rho = -0.07", "Replicated"],
        ["Attribution in reg. regions", "Promoter >> distal", "Promoter >> distal", "Replicated"],
        ["CRE enrichment", "ATAC FC > 0", "ATAC FC > 0", "Replicated"],
        ["Motif clustering by lineage", "25 cell types", "10 focal types", "Replicated"],
        ["Directed evolution designs", "Completed", "Ready to run", "Next step"],
    ]
    add_table(slide, LEFT_MARGIN, Inches(1.4), CONTENT_W, Inches(3.8),
              rows=8, cols=4, data=data,
              col_widths=[Inches(3.2), Inches(3.0), Inches(2.8), Inches(2.7)],
              highlight_cells={
                  (1, 3): GREEN, (2, 3): GREEN, (3, 3): GREEN,
                  (4, 3): GREEN, (5, 3): GREEN, (6, 3): GREEN,
                  (7, 3): BLUE,
              })


def slide_14_next(prs):
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "Next: directed evolution on 153 cell types, then experimental validation")
    # no accent line per user preference
    add_bullets(slide, LEFT_MARGIN, Inches(1.5), CONTENT_W, Inches(4.5), [
        ("In silico mutagenesis: ",
         "Run gReLU design loop on DanioCell model for all 153 cell types"),
        ("Marker gene analysis: ",
         "AUROC/AUPRC per cell type — predicted vs observed marker specificity"),
        ("Temporal regulatory dynamics: ",
         "Leverage 14 stage groups to study regulatory changes across development"),
        ("Experimental validation: ",
         "Select top designed regulatory elements for MPRA/reporter assay"),
        ("Model release: ",
         "Package weights, scripts, and evaluation pipeline for the team"),
    ])


def slide_15_backup_enrichment(prs):
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "[Backup] 120 hpf is enriched for poor performers; early stages dominate best performers")
    # no accent line per user preference
    add_figure(slide, FIG_DIR / "fig4_timepoint_enrichment.png",
               LEFT_MARGIN, Inches(1.3), width=CONTENT_W)


def slide_16_backup_metrics(prs):
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "[Backup] Full metrics for all 8 DanioCell experiments")
    # no accent line per user preference

    data = [
        ["Experiment", "Group", "Gene Pearson", "Track Pearson"],
        ["pretrained_rep0", "pretrained", "0.381", "0.761"],
        ["pretrained_rep1", "pretrained", "0.371", "0.755"],
        ["pretrained_rep2", "pretrained", "0.377", "0.771"],
        ["pretrained_rep3", "pretrained", "0.375", "0.762"],
        ["random_seed42", "random", "0.228", "0.496"],
        ["random_seed43", "random", "0.228", "0.539"],
        ["random_seed44", "random", "0.227", "0.543"],
        ["random_seed45", "random", "0.234", "0.541"],
    ]
    add_table(slide, LEFT_MARGIN, Inches(1.4), CONTENT_W, Inches(4.0),
              rows=9, cols=4, data=data,
              col_widths=[Inches(3.8), Inches(2.2), Inches(2.8), Inches(2.9)],
              highlight_cells={
                  (1, 2): GREEN, (2, 2): GREEN, (3, 2): GREEN, (4, 2): GREEN,
                  (1, 3): GREEN, (2, 3): GREEN, (3, 3): GREEN, (4, 3): GREEN,
                  (5, 2): RED, (6, 2): RED, (7, 2): RED, (8, 2): RED,
                  (5, 3): RED, (6, 3): RED, (7, 3): RED, (8, 3): RED,
              })


def main():
    prs = new_presentation()

    slide_01_title(prs)
    slide_02_takeaway(prs)
    slide_03_dataset(prs)
    slide_04_setup(prs)
    slide_05_training(prs)
    slide_06_headtohead(prs)
    slide_07_comparison(prs)
    slide_08_developmental(prs)
    slide_09_conservation(prs)
    slide_10_attribution(prs)
    slide_11_cre(prs)
    slide_12_motifs(prs)
    slide_13_summary(prs)
    slide_14_next(prs)
    slide_15_backup_enrichment(prs)
    slide_16_backup_metrics(prs)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Saved: {OUT}")
    print(f"Slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
