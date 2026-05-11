# Pre-Publication Repository Review

**Branch:** `code-review` (5 commits ahead of `main`)
**Date:** 2026-05-09
**Reviewers (synthesized lenses):** Avantika Lal (upstream Decima author) + Loic Royer (PI)
**Methodology:** byte-level git-blob comparison against upstream HEAD for each forked tree
- Upstream `decima` HEAD: `d04961e` ("copied code") at `~/github_repos/decima`
- Upstream `decima-applications` HEAD: `d39cc08` ("finish uploading preprint code") at `~/github_repos/decima-applications`
- Comparison data: `/tmp/claude/pubreview/{decima,dapps}_class.tsv`

---

## Executive summary

| # | Finding | Severity |
|---|---|---|
| 1 | **License conflict at the top level** — top-level `LICENSE` is BSD-3-Clause (CZ Biohub), but both `decima-main/` and `decima-applications-main/` are Genentech Non-Commercial. The top-level BSD is misleading. | 🔴 BLOCKER |
| 2 | **Modified files lack required change-notices** — Genentech NC §4 *requires* "prominent notices stating You changed the files". Our 9 modified files in `decima-main` and 4 in `decima-applications-main` carry no such notice. | 🔴 BLOCKER |
| 3 | **49 pristine upstream notebooks republished verbatim** in `decima-applications-main/` (out of 53 upstream files). Two whole directories (`7_eqtls/`, `8_disease/`) are wholly pristine. | 🟠 HIGH |
| 4 | **37 of 72 "NEW" `.py` files in `decima-applications-main/` are jupytext exports of pristine upstream notebooks** — not novel work. | 🟠 HIGH |
| 5 | **WandB API key still in git history** (per memory). Even though scrubbed from current files, history retains it. | 🟠 HIGH |
| 6 | `decima-main/setup.cfg` has empty `install_requires` → repo cannot be installed reproducibly. | 🟡 MED |
| 7 | Hardcoded absolute paths to four users' homes scattered across the codebase. | 🟡 MED |
| 8 | Zero tests (`tests/conftest.py` is the empty stub from upstream). | 🟡 MED |
| 9 | Known live bugs: `loss.py` `eps`, `lightning.py` unreachable `invert` branch, `00_evolve_combined.py` early-stopping window. | 🟡 MED |
| 10 | No `CITATION.cff`; no `NOTICE` file; no `FORK_NOTES.md` documenting upstream provenance. | 🟡 MED |

**Overall verdict:** Not yet ready for public release. Findings 1–2 are non-negotiable for license compliance and must be fixed before any flip to public. Findings 3–4 are reputation risks — Avantika Lal would notice immediately. Findings 5–10 are routine pre-submission hygiene.

---

## 1 · Inventory: `decima-main/`

**Total files:** 49 (excluding `__pycache__/`, `.ipynb_checkpoints/`, `.git/`, `build/`, `dist/`, `*.egg-info/`)

| Category | Count | % |
|---|---|---|
| **PRISTINE** (byte-equal to upstream HEAD) | 29 | 59% |
| **MODIFIED** (changed by us) | 9 | 18% |
| **NEW** (our additions) | 11 | 22% |
| **DELETED** (in upstream, missing here) | 0 | 0% |

### Modified files (with diff sizes vs upstream HEAD)

| File | +/- lines | Substantive? | Change-notice in file? |
|---|---|---|---|
| `src/decima/decima_model.py` | +245 / -32 | YES — substantial (incl. our cherry-pick of HF migration) | ❌ |
| `src/decima/lightning.py` | +236 / -33 | YES — substantial (incl. CUBLAS env var) | ❌ |
| `src/decima/read_hdf5.py` | +147 / -0 | YES — pure additions (likely new dataset class) | ❌ |
| `scripts/finetune.py` | +41 / -12 | YES — moderate (sys.path fix + adapt) | ❌ |
| `src/decima/loss.py` | +5 / -3 | tiny tweak (likely the Poisson/multinomial split commit) | ❌ |
| `src/decima/visualize.py` | +2 / -2 | trivial | ❌ |
| `src/decima/evaluate.py` | +1 / -1 | trivial (the np.nan tuple fix we cherry-picked) | ❌ |
| `scripts/predict_genes.py` | +1 / -1 | trivial (sys.path fix) | ❌ |
| `tutorials/tutorial.ipynb` | +2011 / -553 | LIKELY just re-execution outputs (large diff, but cells probably unchanged) | ❌ |

**License action required:** every modified file needs a prominent notice at the top stating it was modified, per Genentech NC §4. Suggested header:
```
# This file was modified from the original Genentech/decima source by
# Chan Zuckerberg Biohub on 2026-XX-XX. Modifications are documented in
# FORK_NOTES.md. Original copyright Genentech, Inc., 2024 (Genentech
# Non-Commercial Software License v1.0).
```

### New files (ours-only, 11 total)

| File | Purpose |
|---|---|
| `src/decima/lightning_temporal.py` | Temporal-model variant of LightningModel |
| `src/decima/genome.py` | Genome-handling utilities (zebrafish-specific?) |
| `scripts/decima_finetune.py` | Fine-tuning wrapper |
| `scripts/decima_predictions.py` | Prediction wrapper |
| `scripts/finetune_temporal.py` | Temporal fine-tuning entry point |
| `scripts/predict_genes_temporal.py` | Temporal prediction entry point |
| `scripts/submit_decima_finetune.sh` | SLURM submit |
| `scripts/submit_decima.sh` | SLURM submit |
| `scripts/README_decima_finetune.md` | Fine-tuning docs |
| `scripts/results_celltype_models.ipynb` | Results notebook |
| `tutorials/tutorial.py` | Jupytext export (suspect — pristine .ipynb partner exists?) ⚠️ |

### Pristine upstream files (29 total) — candidates for treatment

The 29 pristine files include all of upstream's docs/Sphinx config, package metadata, and 6 of the 14 `src/decima/*.py` files. Specifically:

- **Pristine `src/decima/`**: `__init__.py`, `interpret.py`, `metrics.py`, `preprocess.py`, `variant.py`, `write_hdf5.py`
- **All of `docs/`**: 9 files, byte-identical to upstream
- **Package metadata**: `setup.cfg`, `setup.py`, `pyproject.toml`, `README.md`, `README.rst`, `LICENSE.txt`, `AUTHORS.rst`, `CHANGELOG.rst`, `CONTRIBUTING.rst`, `tox.ini`, `__init__.py`, `fig1.png`
- **`tests/conftest.py`** — the empty upstream stub

These don't need to be modified — they're correctly attributed to Genentech via their own copyright headers and the Genentech NC license. **But we should be honest that almost none of `decima-main/` is our work** — only the 9 modified files and 11 additions represent our contribution.

---

## 2 · Inventory: `decima-applications-main/`

**Total files:** 125 (excluding pycache/checkpoints)

| Category | Count | % |
|---|---|---|
| **PRISTINE** (byte-equal to upstream HEAD) | 49 | 39% |
| **MODIFIED** (changed by us) | 4 | 3% |
| **NEW** (our additions) | 72 | 58% |
| **DELETED** | 0 | 0% |

⚠️ **Sharper truth**: of the 72 "NEW" `.py` files, **37 are jupytext exports of pristine upstream `.ipynb` files** (same basename, `.ipynb` is byte-equal to upstream HEAD). These are NOT novel work — they're auto-generated text exports of upstream's notebooks. So genuinely novel additions are only **35 files**, not 72.

### Modified files (only 4!)

| File | +/- lines | Nature |
|---|---|---|
| `notebooks/4_evaluation/01_evaluate.ipynb` | +93 / -15 | Likely zebrafish-specific evaluation logic |
| `notebooks/4_evaluation/00_predict.ipynb` | +20 / -32 | Adapted for our predictions |
| `notebooks/3_training/01_finetune.ipynb` | +13 / -14 | Minor tweaks to upstream training notebook |
| `notebooks/6_cell_states/modisco_simple.py` | +5 / -1 | Minor |

Each of these 4 needs a change-notice at the top of the file (in a markdown cell for `.ipynb`, in a header comment for `.py`). The `00_predict.ipynb` deletes 32 lines and adds 20 — that's a net decrease, suggesting we removed upstream-specific scaffolding, not added zebrafish logic.

### Pristine upstream files (49 total) — the biggest red flag

**Whole directories that are byte-identical to upstream:**
- `7_eqtls/` — all 5 ipynb + 3 `.py` library files (8 files total) are pristine
- `8_disease/` — all 3 ipynb + 2 `.py` library files (5 files total) are pristine

**Mostly-pristine directories (zebrafish-relevant work appears to be elsewhere):**
- `0_sc_data_prep/` — 5 `.ipynb` files pristine; only `zf-prep.{ipynb,py}` is novel
- `1_processing/` — 5 `.ipynb` + `scimilarity.py` pristine; we added 6 `.py` exports of those pristine ones
- `5_specificity/` — 4 `.ipynb` pristine; we added new `.py` analysis scripts
- `6_cell_states/` — 7 `.ipynb` + 3 `.py` library files pristine
- `9_design/` — 2 `.ipynb` (`0_evolve.ipynb`, `1_read.ipynb`) pristine

**Top-level `LICENSE.txt` and `README.md`** are pristine — i.e., the README still describes upstream's project, not ours. The `LICENSE.txt` is correctly retained, but the README needs to be either replaced or supplemented with one describing our fork.

### Genuinely novel zebrafish work (~35 files)

Filtering out jupytext-of-pristine duplicates, the genuinely novel contributions are:

| Area | Files |
|---|---|
| **Zebrafish data prep** | `0_sc_data_prep/zf-prep.{ipynb,py}` |
| **Zebrafish dataset builder** | `2_dataset/zebrafish.{ipynb,py}`, `zebrafish_data_exploration.ipynb` |
| **Zebrafish evaluation** | `4_evaluation/01_evaluate_celltypes.ipynb`, `00_submit_predict_decima.sh`, `ckpt_dirs_celltypes.txt`, `README_daniodecima_predictions.md` + `00_predict.py`, `01_evaluate.py` (adapted .py exports) |
| **Combined attribution analysis** | `5_specificity/00_combined_attribution_analysis.py`, `00_submit_combined_attributions.sh`, `ensemble_orthologies.ipynb`, `README_attribution_analysis.md` |
| **Cell-type motif attribution** | `6_cell_states/celltype_motif_attribution.py`, `environment_modisco.yml`, `environment_pytorch.yml`, `submit_celltype_motifs.sh`, `README_analysis_TF-MoDISco.md` |
| **Directed-evolution design (zebrafish)** | `9_design/00_evolve_combined.py`, `00_submit_evolve_combined.sh`, `02_summarize_results*.py` (4 variants), `02_submit_summarize*.sh` (3 variants), `1_read_celltypes.py`, `designs_clustering_analysis.ipynb`, `designs_volcano_plot.ipynb`, `README_design.md` |

That's the contribution to highlight in the paper.

### Suspect file pairs (`.py` is jupytext export of pristine upstream `.ipynb`)

37 such pairs exist. Examples:
- `0_sc_data_prep/{bca,heart,lung,retina,skin}-prep.py` — upstream's data prep, jupytext-exported
- `7_eqtls/{0..5}_*.py` — entire upstream eQTL pipeline, jupytext-exported
- `8_disease/{0_overall,1_modisco,2_examples}.py` — entire upstream disease pipeline

**Decision needed:** drop these `.py` exports? They duplicate the pristine `.ipynb` files (which are themselves candidates for deletion).

---

## 3 · Licensing audit

### The conflict
- **Top-level `/LICENSE`**: BSD 3-Clause, © 2023 Chan Zuckerberg Biohub.
- **`decima-main/LICENSE.txt`**: Genentech Non-Commercial Software License v1.0.
- **`decima-applications-main/LICENSE.txt`**: Genentech Non-Commercial Software License v1.0.

A reader cloning the repo will see the BSD-3 at top and assume the whole repo is permissively licensed. But the contained Genentech NC license **prohibits commercial use** — including, per its broad §1 definition, "use of the Work in the discovery, research, pre-clinical or clinical development, or related manufacturing of diagnostic, prognostic, prophylactic, or therapeutic treatments". That's a meaningful restriction the BSD-3 wrapper hides.

### What Genentech NC §4 (Redistribution) requires us to do

We are redistributing modified Genentech code, so we must:

1. ✅ **Give recipients a copy of the Genentech NC license** — done (license files retained in both forks).
2. ❌ **"Cause any modified files to carry prominent notices stating that You changed the files"** — *not* done. The 9 modified files in `decima-main/` and 4 in `decima-applications-main/` lack such notices.
3. ✅ **Retain copyright/patent/trademark/attribution notices from upstream** — appears done; upstream copyright headers are intact in pristine files.
4. ⚠️ **If upstream has a NOTICE file, redistribute it** — upstream does not appear to have a NOTICE file (none in `~/github_repos/decima` or `~/github_repos/decima-applications`), so no action needed here.

### Recommended structure

Adopt one of these patterns:

**Option A — Repo-wide dual-license disclosure (simplest)**
- Replace top-level `/LICENSE` with a `LICENSING.md` that says: "This repository combines code under two licenses: (1) `decima-main/` and `decima-applications-main/` are derived from Genentech's Decima and decima-applications and remain under the Genentech Non-Commercial Software License v1.0 (see those subdirs' `LICENSE.txt`); (2) all original CZ Biohub contributions outside those subdirs, and the modifications recorded in `FORK_NOTES.md`, are licensed BSD-3-Clause (see `LICENSE.bsd`)."
- Add `NOTICE` at top level acknowledging Genentech, Decima, and decima-applications.
- Add `FORK_NOTES.md` per fork enumerating modified files and the nature of each modification.

**Option B — Upstream-as-dependency (cleanest, more work)**
- Strip `decima-main/` to only the modified + new files, structured as overlay/extension package.
- Pin upstream `decima` as a Git+commit dependency in `setup.cfg`.
- Same for `decima-applications-main/`: keep only zebrafish-novel notebooks/scripts; document that users should clone upstream separately for the comparison notebooks.
- Result: this repo carries only our novel work + thin overlay = clearer story, smaller fork, easier license accounting.

I recommend Option A for speed, Option B if there's appetite. The user should decide.

---

## 4 · Critical review: Avantika Lal lens (upstream Decima author)

If I were Avantika Lal browsing `czbiohub-sf/DanioDecima` after it goes public, here's what I'd notice in the first 5 minutes:

1. **"Why is most of my code in here verbatim?"** Two whole subdirectories of decima-applications (`7_eqtls/`, `8_disease/`) are byte-identical copies of my work. There is no zebrafish-specific eQTL or disease analysis here — yet you've republished my notebooks. **A fork should contain your novelty plus the modifications you needed; pristine upstream files should be removed and the upstream pinned as a dependency, or at minimum, the README should clearly say "these dirs are unchanged from upstream and are kept only so the data-prep pipeline runs end-to-end".**

2. **"Where are your change-notices?"** The Genentech license requires modified files carry a notice. Your `decima_model.py` adds 245 lines of new code with no `# Modified from Genentech/decima…` header. That is a license-compliance miss, not a stylistic one.

3. **"What's actually new here?"** From the file listing alone I can tell what's novel — zebrafish data prep, the directed-evolution work, the combined attribution analysis. But neither the top-level README nor the per-directory READMEs lead with that. A reviewer should land on this repo and see in 30 seconds: *here is the novelty, here is what was adapted, here is what was kept unchanged for reproducibility*. Currently they have to do the diff themselves.

4. **"Why are jupytext exports of my notebooks committed?"** 37 `.py` files in your applications fork are mechanical jupytext exports of my pristine `.ipynb`s. They duplicate the notebooks for no obvious reason. If you converted them so reviewers could see line-numbered diffs, fine — but say so, and don't ship them in the public repo.

5. **"Did you cite our Decima paper?"** Top-level `README.md` cites the Decima preprint correctly (`lal2024decoding`) ✅ and includes a DanioDecima BibTeX placeholder. No `CITATION.cff` yet — add one so GitHub's "Cite this repository" button works.

**Avantika's likely verdict:** "This is sloppy in the way that suggests you copied the repo, made minimal upstream changes, and bolted your zebrafish work on top. Clean up the pristine-republishing problem and add change-notices and I'll have no objection. As-is, I'd push back on this in a review."

---

## 5 · Critical review: Loic Royer lens (PI / publication readiness)

If I were Loic looking at this two days before submission:

1. **"Can I `pip install` this in a clean environment?"** No — `decima-main/setup.cfg` has empty `install_requires`. We have ~15 undeclared deps. The README would tell a reviewer to run `pip install .` and it would silently succeed but break at import time.

2. **"Are there secrets in the git history?"** Per memory, the WandB API key was scrubbed from working files but **remains in git history**. Before flipping public, either rotate the key (if not already done) AND leave history, or run `git filter-repo` to scrub. Rotation is the lower-risk path.

3. **"Is the science reproducible from a clean checkout?"** The 150+ hardcoded paths to `/hpc/projects/data.science/...` and four users' homes mean nobody outside our cluster can run anything without manual edits. Either parameterize via a `config.yaml` / env vars, OR add a clearly-marked `# TODO: edit this for your environment` comment on every such path AND a top-level README section listing them.

4. **"Where are the tests?"** Zero. `tests/conftest.py` is the empty upstream stub. We don't need exhaustive coverage, but **at least one smoke test** that loads a checkpoint and runs a forward pass would protect us against trivial regressions.

5. **"What's the publishable contribution?"** Top-level `README.md` *is* well-written and DanioDecima-framed (architecture, workflows, usage, citations for Decima + ZebraHub + DanioDecima). It already includes a Pre-Submission Checklist that overlaps much of what I'd ask for. Two issues remain: (a) it ends with `License: BSD-3-Clause license`, which contradicts the contained Genentech-NC code (the **license conflict** in §3); (b) `decima-applications-main/README.md` is the pristine upstream one-line README — replace with our own framing.

6. **"Is the model accessible?"** The `model-checkpoints` branch holds checkpoints in git. That's a reproducibility problem (cloners pull big binaries) and possibly a public-repo size problem. Move to Zenodo or HuggingFace before going public; reference from README. (You said leave that branch alone — fine, but worth noting.)

7. **"Does the paper cite this repo and vice versa?"** Need `CITATION.cff` here, and the manuscript needs a Code Availability section pointing at the specific commit/tag we publish.

**Loic's likely verdict:** "Fix the install path, scrub or rotate the keys, get the README right, and put one smoke test in. Then we can flip it public."

---

## 6 · Decision points (before any fixes get committed)

1. **Licensing structure** — Option A (repo-wide dual-license disclosure) or Option B (strip forks down to overlays + pin upstream)?
2. **Pristine upstream files in `decima-applications-main/`** — delete (cleanest), or keep with clear "unchanged from upstream" docs (more conservative)?
3. **Jupytext `.py` exports of pristine `.ipynb`s** — delete the 37 of them?
4. **WandB key in history** — rotate key only, or also `git filter-repo` scrub history? (Scrub is destructive and changes commit hashes.)
5. **Hardcoded paths** — parameterize via config (proper fix), or document in README and leave (faster)?
6. **Smoke test** — add a minimal one before public, or accept tests-debt and ship?
7. **Top-level README rewrite** — happy for me to draft a paper-style README that opens with the novelty story, lists components by license, and includes citation/install/quickstart?
8. **`tutorials/tutorial.ipynb`** — the +2011/-553 diff: should I confirm this is just re-execution outputs (pristine logic) and offer to revert to upstream's clean version, or keep our re-executed copy?

---

## 7 · Recommended punch list (prioritized)

### P0 — license-compliance blockers (cannot flip public until these are done)

- [ ] Add change-notices to all 13 modified files (9 in `decima-main/`, 4 in `decima-applications-main/`)
- [ ] Restructure top-level licensing: replace `/LICENSE` with `LICENSING.md` + add `NOTICE` + add `FORK_NOTES.md` per fork
- [ ] Rotate the leaked WandB API key (and confirm rotation happened)

### P1 — reputation-risk fixes

- [ ] Decide and execute on the 49 pristine upstream files in `decima-applications-main/` (delete or document)
- [ ] Decide and execute on the 37 jupytext-of-pristine `.py` files (delete recommended)
- [ ] Rewrite top-level `README.md` to lead with novelty, attribution, and quickstart
- [ ] Replace `decima-applications-main/README.md` (currently pristine upstream) with one that says "see top-level README; this directory is a fork of Genentech/decima-applications"

### P2 — reproducibility

- [ ] Populate `decima-main/setup.cfg`'s `install_requires` from actual imports
- [ ] Address hardcoded paths (parameterize OR document prominently)
- [ ] Add `CITATION.cff` at top level (cite both Decima paper and our forthcoming preprint)
- [ ] Add at least one smoke test (load checkpoint → forward pass → no exception)

### P3 — quality / known bugs

- [ ] Fix `loss.py` `eps` (defined but unused; div-by-zero/log-zero unprotected)
- [ ] Fix `lightning.py:666-671` unreachable `invert` branch
- [ ] Clarify `00_evolve_combined.py:262-272` early-stopping semantics (or fix if buggy)
- [ ] Strip `sys.path` hacks in `lightning.py:26` and `lightning_temporal.py:23` (hardcoded `/code/decima/`)
- [ ] Optional: scrub WandB key from git history with `git filter-repo` (rewrites all hashes)

---

## Appendix A — Data files

Classification TSVs (path = `/tmp/claude/pubreview/`):
- `decima_class.tsv` — every file in `decima-main/`, classified
- `dapps_class.tsv` — every file in `decima-applications-main/`, classified
- `decima_upstream.tsv`, `decima_fork.tsv` — raw hash manifests for decima
- `dapps_upstream.tsv`, `dapps_fork.tsv` — raw hash manifests for decima-applications

These are working scratch — not committed. If we want them as evidence, copy into `documents/`.

## Appendix B — `code-review` branch state at time of review

Commits ahead of `main` (5):
```
4101894 Fix CUBLAS_WORKSPACE_CONFIG for deterministic training on CUDA >= 10.2
dfdf3bd Fix sys.path shadowing: use insert(0) not append for local decima modules
bccf43d Fix tuple unpack crash in compute_marker_metrics
d4e0212 Migrate Borzoi weight loading from WandB to HuggingFace
d3058e9 Remove exposed WandB API keys and add pre-submission checklist
```

Top 4 are cherry-picks from `daniocell-pipeline` (publication-relevant decima-core fixes; daniocell-only files omitted from each pick).
