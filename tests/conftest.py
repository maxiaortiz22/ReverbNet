"""
Pytest configuration.

Mirrors ``run.py``: the repository root and the ``code/`` directory are put on
``sys.path`` so the package and its absolute ``from filters import ...`` import
resolve. This workaround goes away once the package is renamed and made
installable (CODE-04 / BUG-08).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for path in (ROOT, ROOT / "code"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
