#!/usr/bin/env python3
"""Preview: 3 representative slides for design feedback."""

import sys
from pathlib import Path

_helpers_dirs = [
    *Path.home().glob(".claude/plugins/marketplaces/*/plugins/generate-slides/scripts"),
    *Path.home().glob(".claude/plugins/cache/*/generate-slides/*/scripts"),
]
for d in _helpers_dirs:
    if (d / "pptx_helpers.py").exists():
        sys.path.insert(0, str(d))
        break

from pptx_helpers import *

FIG_DIR = Path("/hpc/scratch/group.data.science/yang-joon.kim/"
               "daniodecima-daniocell/02_evaluation/manuscript_figures")
OUT = Path("/hpc/projects/data.science/yangjoon.kim/step/docs/slides/"
           "daniocell-preview.pptx")


def slide_content_bullets(prs):
    """Slide 2: content + bullets (tests typography, spacing, emphasis)."""
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "We validated a fine-tuning framework on Zebrahub — can it scale to richer data?")
    add_accent_line(slide)
    add_bullets(slide, LEFT_MARGIN, Inches(1.5), CONTENT_W, Inches(2.8), [
        ("DanioDecima (Zebrahub): ",
         "Pretrained >> random, conservation-independent, biologically meaningful attributions"),
        ("Key question: ",
         "Does the same framework work with a different, richer atlas (DanioCell)?"),
        ("DanioCell advantage: ",
         "3.4x more pseudobulks, 2.5x denser cell-type x stage coverage, 14 stage groups"),
    ])
    add_callout_box(slide, LEFT_MARGIN, Inches(4.8), CONTENT_W, Inches(0.8),
                    "This talk: ",
                    "We show that all DanioDecima findings replicate with DanioCell — the framework generalizes",
                    font_size=18)


def slide_figure(prs):
    """Slide 7: figure-heavy (tests figure sizing, placement)."""
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "80% of test genes are better predicted by the pretrained model (MWU p=0.014)")
    add_accent_line(slide)
    add_figure(slide, FIG_DIR / "fig3_model_comparison.png",
               LEFT_MARGIN, Inches(1.2), height=Inches(5.8))


def slide_table(prs):
    """Slide 6: table comparison (tests table styling)."""
    slide = add_blank_slide(prs)
    make_content_slide(slide,
        "Gene-level Pearson is identical (0.38); pretrained lift is actually larger (+0.23 vs +0.19)")
    add_accent_line(slide)
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


def main():
    prs = new_presentation()
    slide_content_bullets(prs)
    slide_figure(prs)
    slide_table(prs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Preview saved: {OUT}")
    print(f"Slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
