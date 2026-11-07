---
name: cpp-code
description: Conventions for ReverbNet's C++ code in code/cpp/ (C++17, pybind11, CMake, the audio_processing extension). Use when creating, modifying, building or binding C++ code, when touching bindings.cpp or CmakeLists.txt, or when deciding whether something is worth doing in C++ instead of numpy.
---

# C++ code in ReverbNet

## Before writing C++: is it needed?

Today the extension (`audio_processing`) only exposes `AudioProcessor` (rms, snr, rms_comp),
`ClarityCalculator` and `DefinitionCalculator`, each one or two lines of numpy. Its future is
under discussion (CODE-03). Only add new C++ when:

1. profiling shows that numpy/scipy is not enough for something on the critical path, or
2. the goal is real-time inference / a plugin (RES-04).

Otherwise implement it in Python (`python-code` skill) and record the decision.

## Layout

```
code/cpp/
├─ CmakeLists.txt            # pybind11_add_module(audio_processing ...)
├─ bindings.cpp              # the single binding entry point (PYBIND11_MODULE)
├─ audio_processor.{hpp,cpp}
├─ clarity/clarity_calculator.{hpp,cpp}
├─ definition/definition_calculator.{hpp,cpp}
├─ audio_processing.py       # loader that looks for the .pyd under build/
└─ pybind11/                 # submodule
```

- One component = one folder with `.hpp` + `.cpp`. Add the `.cpp` to `pybind11_add_module`
  and the folder to `target_include_directories`.
- Bindings live only in `bindings.cpp`; the C++ core does not include pybind11 headers (so it
  can be reused in a plugin or in native tests).

## Style

- C++17. `PascalCase` classes, `snake_case` methods and variables (matching Python).
- Doxygen comments (`@brief`, `@param`, `@return`, `@throws`) in the header, with units.
- `const&` for inputs; no unnecessary copies. `std::size_t` for indices.
- No `using namespace std;` in headers. Project includes before system includes.
- Format with `clang-format` (style to be defined in a `.clang-format` if adopted).

## Bindings (pybind11)

- Take arrays without copying: `py::array_t<double, py::array::c_style | py::array::forcecast>`
  and access them with `arr.unchecked<1>()`. The current `numpy_to_vector` helper copies the
  buffer: do not use it in new code.
- Validate `ndim` and sizes in the binding; throw `std::invalid_argument` (surfaces in Python
  as `ValueError`) or `std::runtime_error`.
- In long loops that touch no Python objects, release the GIL with `py::gil_scoped_release`.
- Return `py::array_t` for arrays and `double` for scalars.

## Numerical robustness

- Never index out of range: if an index derived from `fs`/time exceeds `size()`, clamp it or
  throw (today `ClarityCalculator` assumes `t < size()`).
- Guard divisions and `log10` with an epsilon, as in Python.
- Accumulate energies in `double`.
- Never assume the direct sound is at index 0 (BUG-02): callers pass the RIR already trimmed
  with `trim_to_onset`, or new functions take the onset index as an explicit parameter.

## Equivalence with Python

- Every C++ function has a numpy reference implementation and a test comparing both with
  `np.testing.assert_allclose` (rtol ~1e-12) on synthetic signals.

## Build and test

```bash
git submodule update --init --recursive
python build.py                      # deletes code/cpp/build and builds in Release (Windows)
python scripts/test_compilation.py
```

- The `.pyd` ends up at `code/cpp/build/Release/audio_processing.cp312-win_amd64.pyd`; it must
  match the env's Python version (`cp312`). Build inside the `ReverbNet` env.
- `code/cpp/audio_processing.py` looks for the `.pyd` in fixed paths; if the name or the
  platform changes, update that list.

## If the goal is real time (RES-04)

- No allocations, locks or I/O in the audio callback; reserve memory in `prepare()`.
- Block-based processing with explicit state (IIR filters keep their state).
- Cover with tests that block-based processing matches offline processing.
