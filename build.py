"""
Lightweight build helper for the project’s C++ extension (Windows workflow).

This script deletes any existing local CMake build directory, regenerates the
build system, compiles in **Release** configuration, and then copies the
resulting ``.lib`` artifacts up into ``code/cpp`` so they can be imported from
Python (via the accompanying extension module).

Usage
-----
Run directly:

    python build.py

Notes
-----
- Paths are constructed relative to the current working directory at runtime.
  Make sure you invoke this script from the project root so the relative paths
  resolve correctly.
- The script calls CMake through ``os.system``; no error checking is performed.
  Inspect the console output for build failures.
"""

import os
from shutil import copy
from glob import glob
from pathlib import Path
import shutil


def compile_library():
    """
    Configure and build the C++ library using CMake (Release config).

    Steps
    -----
    1. Record the current working directory (assumed project root).
    2. Change into ``code/cpp``.
    3. Delete any pre‑existing ``build`` directory to ensure a clean build.
    4. Create a fresh ``build`` directory and ``cd`` into it.
    5. Run CMake configure & build in Release mode.
    6. Copy generated ``.lib`` files from ``Release/`` back into ``code/cpp``.
    """
    # Compile the library.
    main_dir = os.getcwd()
    os.chdir(main_dir + '/code/cpp')

    # Delete the build folder if it exists.
    if os.path.exists('build'):
        shutil.rmtree('build')

    # Create the build folder.
    os.makedirs('build', exist_ok=True)
    os.chdir('build')

    # Configure & build with CMake (Release configuration).
    os.system("cmake ..")
    os.system("cmake --build . --config Release")

    # Copy the resulting library (.lib) files to the cpp directory.
    for file in glob('Release/*.lib'):
        copy(file, main_dir + '/code/cpp')


if __name__ == '__main__':
    """Build script entry point (Windows)."""
    compile_library()