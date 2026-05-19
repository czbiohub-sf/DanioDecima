# Pre-Submission Checklist

> **Status (2026-05-18):** Most items have been resolved during publication-prep
> (see `documents/publication_review.md` and the `code-review` branch history).
> This file is preserved as an audit trail. Items still requiring action are
> marked **TODO**; resolved items are marked with `[x]` and a brief note.

## Security

- [x] **Rotate WandB API keys** — Keys (`66d3a7...` and `f4888f01...`) were
  scrubbed from the working tree in commit `d3058e9` and physically removed
  from git history via `git filter-repo` in Wave 7 of publication-prep. The
  user will rotate the corresponding WandB-dashboard credentials before flipping
  the repo public. (entity `czbsf-comp-bio`, project `decima-zebrafish`)

## Reproducibility

- [ ] **TODO — Remove hardcoded paths:** ~150+ hardcoded absolute paths across
  the codebase still reference user-specific HPC directories
  (`/hpc/mydata/mathias.voges/`, `/home/karollua/`, `/home/gunsalul/`,
  `/hpc/scratch/.../yangjoon.kim/`, `/gstore/data/resbioai/`, `/code/decima/`).
  Replace with environment variables, config files, or CLI arguments. Priority
  files:
    - Core library: `src/decima/decima_model.py:109`, `lightning.py:26`,
      `lightning_temporal.py:23`
    - Scripts: `finetune.py:15`, `finetune_temporal.py:15`,
      `decima_finetune.py:44,167`
    - Shell: `submit_decima.sh:25-26`, `submit_decima_finetune.sh:4-7`
    - Notebooks: all directories (`0_sc_data_prep` through `9_design`)
  *Note: accepted as-is per user decision for the preprint; documenting for
  future cleanup.*
- [ ] **TODO (partial) — sys.path hacks:** The `sys.path.insert(0, ...)` ordering
  fix landed in Wave 0, but the hardcoded `/code/decima/src/decima/` paths in
  `lightning.py:26` and `lightning_temporal.py:23` remain.
- [x] **Populate `install_requires`** — Done in Wave 9 (commit `2a5bd72`);
  19 direct deps derived from imports. `environment.yml` also added for
  exact reproducibility.
- [ ] **TODO — Standardize seed handling** across scripts.

## Correctness

- [ ] **TODO — Loss function epsilon:** `loss.py` defines `self.eps` but never
  uses it in `forward()` — division by zero and `log(0)` are unprotected
  (lines 34-35).
- [x] **Unreachable code: `lightning.py:get_task_idxs`** — Fixed in the
  CodeQL-quality commit (`662806a`); the `invert` parameter was removed
  along with the dead block.
- [ ] **TODO — Early stopping overlap:** `00_evolve_combined.py:262-272` —
  comparison windows may overlap.

## Code Quality

- [ ] **TODO — Add test suite** (currently zero tests; `tests/conftest.py`
  is a stub). Not a public-flip blocker; a single smoke test would be a
  nice next step.
- [ ] **TODO — Remove dead code:**
    - `02_summarize_results_designs_test.py` (~1200 commented lines)
    - `lightning.py:281-341` (commented temporal-smoothness block)
- [x] **Bare `except` clauses fixed:** `setup.py:13` (now `except Exception`)
  and `preprocess.py:228` (now narrow tuple) — commit `662806a`.
- [x] **Package metadata updated:** `setup.cfg` description and URL set to
  the DanioDecima fork (commit `2a5bd72`).

## Documentation

- [ ] **TODO — Update citation:** Author list is "Voges, Mathias, et al." —
  needs full author list, venue, and year. Fix locations:
    - `README.md:124-129` (DanioDecima BibTeX entry)
    - `NOTICE` line 37 (DanioDecima entry)
  The `NOTICE` file also has two `[bioRxiv ID]` placeholders for the upstream
  Decima (line 10) and Borzoi (line 22) DOIs that should be filled in.
- [ ] **TODO — Fix path references** in
  `documents/DanioDecimaPipeline_README.md:209-237` (13 paths point to wrong
  user directory).
- [ ] **TODO — Fix missing environment file:** `README_analysis_TF-MoDISco.md`
  references `environment_pytorch_full.yml`, which does not exist; the actual
  file is `environment_pytorch.yml`.

## CZI open-source compliance (added 2026-05-18)

- [x] `LICENSE.md` — replaced top-level `LICENSE` (commit `f28ecc3`)
- [x] `SECURITY.md` — added (commit `4b812e3`)
- [x] `CONTRIBUTING.md` — added (commit `4b812e3`)
- [x] README: Code of Conduct mention, security reporting, project-status
      declaration — added (commit `4b812e3`)
- [ ] **TODO — Branch protection on `main`** (to be configured by the user
      in repo Settings).

## CodeQL quality alerts

- [x] Error- and Warning-level rules (Modification of parameter with default,
      Unreachable code) — fixed in commit `662806a`.
- [x] 4 Note-level reliability rules (Non-standard exception in special
      method, Except block handles BaseException ×2, Empty except) — fixed
      in commit `662806a`.
- [x] Unused import (61 of 61) — auto-fixed via `ruff F401` in commit
      `662806a`.
- [ ] **TODO — 5 remaining Note-level Maintainability rules:**
      commented-out code (6), unused local variable (6), statement has no
      effect (4), mixed explicit/implicit returns (1), unused global
      variable (1). Tracked for a separate cleanup commit.
