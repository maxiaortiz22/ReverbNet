---
name: python-code
description: Conventions for writing and modifying Python code in ReverbNet (the code/ package, future reverbnet/, run.py, scripts/, configs/). Use whenever a .py file is created or edited - style, type hints, docstrings, error handling, randomness, logging, multiprocessing on Windows, imports and tests.
---

# Python code in ReverbNet

## Style

- Python 3.12. Code, names, docstrings and comments in **English**.
- Type hints on every public function (`np.ndarray`, `Sequence[float]`, `Path`, `float | None`).
- **NumPy-style** docstrings (like the existing code): `Parameters`, `Returns`, `Raises`,
  `Notes`. State units (Hz, s, dB, %) and array shapes (`shape (n_bands, n_samples)`).
- Formatting compatible with `ruff format` (line length 100) and `ruff check`. If the
  configuration is added, it goes into `pyproject.toml` (CODE-04).
- `pathlib.Path` for paths; never concatenate strings with `/`.
- f-strings; UPPER_CASE constants at the top of the module.
- Small, pure functions for DSP and acoustics: arrays + parameters in, arrays out. No global
  state, no I/O inside.
- Legacy code with Spanish names (`señal_1`, `ruido_dB`, `DRR_buscado`): do not rename outside
  a backlog item; **new** code is in English.

## Numerics

- Vectorize with numpy; avoid Python loops over samples.
- `float64` to compute descriptors and energies; `float32` to store features and for the model.
- Always guard `log10`: `10 * np.log10(x + EPS)` or `np.maximum(x, EPS)`, with
  `EPS = np.finfo(float).eps` defined once.
- Do not store numeric data as Python lists inside DataFrames (`object` columns): use 2D
  arrays (CODE-02).

## Errors

- `except Exception: continue` / `except: pass` are **forbidden**. Catch only the specific,
  expected failures and **count** each discard by reason (BUG-05). The generator's pattern:
  a `collections.Counter` keyed `'<unit>:<reason>'` (`band:lundeby_noise`,
  `tr_variant:tr_augmentation_<cause>`, `drr_variant:unrealizable`), returned by the worker
  and summarized in `generation_summary.json`. Unexpected exceptions must propagate.
- Domain failures get their own exception class (`NoiseError`, `TrAugmentationError` with
  `.cause`, `DrrAugmentationError`); never wrap programming errors (`NameError`, `TypeError`,
  `AttributeError`, `KeyError`) into them.
- Validate inputs at the top of the function and raise `ValueError` with a clear message.
- Do not use `assert` to validate data at runtime (it is disabled with `-O`); use
  `if ...: raise ValueError(...)`.
- Functions that can fail either return a value or raise; never return `None` as a silent
  error signal.

## Randomness and reproducibility

- **One project-wide seed**: the config's `seed` drives every random process. Never add a
  second seed key to the configs.
- No global `np.random.*` and no `random.seed(time)`. Pass an explicit
  `rng: np.random.Generator` parameter.
- Each random process has its own stream id in `RNG_STREAMS` (`code/random_streams.py`, helper `item_rng`).
  Derive per-item generators from the seed, the stream id and the item identity:
  `np.random.default_rng([seed, RNG_STREAMS['<process>'], zlib.crc32(item_name.encode())])`.
  This keeps draws independent of processing order, multiprocessing and the rest of the file
  list. Use `zlib.crc32`, never `hash()` (salted per process).
- Add a new stream id for a new random process; never reuse or renumber an existing one, and
  never reorder the draws inside a process (both change previously reproducible databases).
- File listings are always sorted: `sorted(path.glob("*.wav"))` (BUG-06).

## Logging and output

- New code uses `logging.getLogger(__name__)`, not `print`. Scripts configure logging in `main()`.
- Progress bars: `tqdm` (preferred) or `progress` (legacy).
- Windows: the console is cp1252. If a script prints non-ASCII characters (ρ, ±, emojis), call
  `sys.stdout.reconfigure(encoding="utf-8")` or avoid them.

## Imports and structure

- Current package: `code/` (shadows the stdlib `code` module, BUG-08; renamed to `reverbnet/`
  in CODE-04). Until then, do not add more `sys.path.append(...)`: use relative imports inside
  the package (`from .filters import BandpassFilter`).
- TensorFlow is imported lazily inside functions (existing pattern in `model.py`) so that data
  generation does not load TF.
- All reusable logic lives in the package; notebooks and scripts import it (CODE-09).

## Multiprocessing (Windows = spawn)

- Entry points guarded with `if __name__ == "__main__":`.
- Everything sent to workers must be picklable (no lambdas, no non-picklable C++ objects);
  rebuild heavy objects inside the worker.
- Return compact results (arrays) rather than huge lists of dicts.

## Tests

- `tests/` with `pytest` (CODE-07). Every new or fixed DSP/acoustics function gets a test with
  a known-result case (see the `dsp` and `room-acoustics` skills).
- Fast tests (< 1 s each) that do not depend on `data/` or `cache/`: generate synthetic signals
  inside the test.
- Run with `python -m pytest -q` from the repo root.

## Checklist before calling it done

- [ ] Type hints and docstrings with units and shapes.
- [ ] No silenced exceptions, no global RNG.
- [ ] Same processing at training and inference time (shared function).
- [ ] New or updated tests, passing.
- [ ] Tracking updated (`improvement-tracking` skill).
