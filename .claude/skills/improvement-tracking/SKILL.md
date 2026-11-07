---
name: improvement-tracking
description: Maintains ReverbNet's improvement tracking in docs/improvements/ (board, backlog, experiments, log). Use when starting or finishing an item (BUG-xx, DATA-xx, TRAIN-xx, CODE-xx, RES-xx), when discovering a new problem or idea, when dropping something, when recording a decision, or when the user asks what is left, what was done or what state the project is in.
---

# Improvement tracking

`docs/improvements/` is the source of truth for the improvement plan. Keeping it up to date
is part of finishing any task, not an optional step.

## Files and responsibilities

| File | Contains | Rule |
| --- | --- | --- |
| `README.md` | Board: ID, title, priority, phase, **status**, branch/PR | The only place where status lives. |
| `backlog.md` | Per-item details: files, problem, evidence, proposal, acceptance | No status; new evidence gets appended. |
| `experiments.md` | One entry per experiment with metrics | Append only; never rewrite an old entry. |
| `log.md` | Chronological log of what was done and decided | Append only; absolute dates. |

## Before starting a task

1. Read the board and check whether the task already has an ID (or overlaps with one).
2. If it does not exist, create the item (see "New item").
3. Set the status to `in progress` and fill in the "Branch / PR" column.
4. If the task depends on another pending item (e.g. retraining before fixing BUG-01), tell
   the user before continuing.

## When finishing

1. Board: status → `done` (or `blocked` / `dropped`), with branch and PR.
2. Backlog: add a **Resolution** line to the item stating what was done and where
   (`file:line`, commit). If the original proposal changed, say so.
3. If the change affects data or metrics: an entry in `experiments.md` (`experiments` skill)
   comparing against the current baseline.
4. `log.md`: an entry dated `YYYY-MM-DD` with what was done, what was decided and the IDs touched.
5. If new problems showed up during the work, create them as items (do not silently fix
   them out of scope).

## New item

- Pick the category: `BUG` (correctness: the current result is wrong), `DATA` (data and
  processing), `TRAIN` (model and training), `CODE` (engineering), `RES` (research).
- ID = next free number in that category, two digits. **Never reuse IDs**, even for dropped items.
- Priority: `P0` invalidates results · `P1` high impact · `P2` improvement · `P3` exploratory.
- Add the row to the board (ordered by ID within its category) and the section to the
  backlog with: **Files**, **Problem**, **Evidence** (measured, not assumed), **Proposal**,
  **Acceptance**.

## Rules

- Evidence must be verifiable: measured numbers, `file:line`, the command that reproduces it.
  If something is a hypothesis, say so.
- Dropping is not deleting: status `dropped` + the reason in the backlog and the log.
- Design decisions (e.g. "we migrate to PyTorch", "we remove the C++ code") go to the log
  with the reason and the alternatives considered.
- Commits and branches mention the ID: branch `fix/BUG-01-causal-convolution`, commit
  `Fix causal convolution alignment (BUG-01)`.
- Keep the board scannable: one-line titles; details go to the backlog.
- Everything is written in English.
