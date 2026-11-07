# Log

Entries in chronological order, most recent **at the end**. Absolute dates
(YYYY-MM-DD). Each entry states what was done, what was decided and which items it touches.

---

## 2026-11-07 — Initial review and workflow setup

- Full review of the code (pipeline, augmentation, Lundeby, model, C++, notebooks) and of
  the thesis. Findings recorded in [backlog.md](backlog.md) with `BUG/DATA/TRAIN/CODE/RES` IDs.
- Checks performed during the review:
  - Onset delay of measured RIRs: great_hall ~33 ms, octagon ~31 ms, classroom ~17 ms
    (median) → evidence for BUG-02.
  - `fftconvolve(mode='same')` starts at `(len(rir)-1)//2` → evidence for BUG-01.
  - `import tensorflow` from the repo root fails because of the `code` package → BUG-08.
  - The `ReverbNet` env (TF 2.16.1, Python 3.12, numpy 1.26.4) sees no GPU → TRAIN-06.
- Computed **baseline v0** from `results/exp1..4` with the new
  `scripts/summarize_results.py` (see [experiments.md](experiments.md)).
- Created branch `chore/project-workflow-setup` with:
  - `CLAUDE.md` (project guide).
  - Skills in `.claude/skills/`: `python-code`, `cpp-code`, `dsp`, `room-acoustics`,
    `neural-network`, `data-analysis`, `experiments`, `improvement-tracking`.
  - This `docs/improvements/` folder.
- Also committed previously uncommitted work from `main`: `analyze_model.py`,
  `calculate_macs_manual.py`, `scripts/model_analysis_simple.py` (model size / MACs
  analysis, related to RES-04) and a change to `notebooks/exp2.ipynb`.
- **Decision**: all repository content (code, docstrings, comments, docs, skills, tracking,
  commit messages) is written in **English** for consistency with the rest of the repo.
