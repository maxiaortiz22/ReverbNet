"""
Dynamic loader for the compiled ``audio_processing`` extension module.

This wrapper searches a handful of common build output locations beneath
the current directory (``build/Release/``, ``build/``, etc.) for the
pybind11-compiled extension (``audio_processing*.pyd`` on Windows). The first
match found is inserted on ``sys.path`` and imported. All public symbols from
the compiled module are then re-exported so downstream code can simply::

    from code.cpp.audio_processor import OctaveFilterBank

If the extension cannot be located, an informative ``ImportError`` is raised
listing the paths that were checked.
"""

import os
import sys
from pathlib import Path

# Directory containing this Python wrapper.
current_dir = Path(__file__).parent

# Candidate locations for the compiled extension (.pyd).
possible_paths = [
    current_dir / "build" / "Release" / "audio_processing.cp312-win_amd64.pyd",
    current_dir / "build" / "audio_processing.cp312-win_amd64.pyd",
    current_dir / "audio_processing.cp312-win_amd64.pyd",
    current_dir / "build" / "Release" / "audio_processing.pyd",
    current_dir / "build" / "audio_processing.pyd",
    current_dir / "audio_processing.pyd",
]

# Try to locate the compiled module at one of the known paths.
module_path = None
for path in possible_paths:
    if path.exists():
        module_path = path
        break

if module_path is None:
    # Fallback: walk the build tree looking for any audio_processing*.pyd.
    build_dir = current_dir / "build"
    if build_dir.exists():
        for root, dirs, files in os.walk(build_dir):
            for file in files:
                if file.startswith("audio_processing") and file.endswith(".pyd"):
                    module_path = Path(root) / file
                    break
            if module_path:
                break

if module_path is None:
    raise ImportError(
        f"Could not locate compiled audio_processing extension. "
        f"Searched: {[str(p) for p in possible_paths]}"
    )

# Ensure the module's directory is importable.
module_dir = module_path.parent
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

try:
    # Import the compiled extension.
    import audio_processing
except ImportError as e:
    raise ImportError(
        f"Error importing compiled audio_processing module from {module_path}: {e}"
    )

# Re-export all public attributes from the compiled module.
__all__ = []
for name in dir(audio_processing):
    if not name.startswith('_'):
        globals()[name] = getattr(audio_processing, name)
        __all__.append(name)