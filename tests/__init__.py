"""Test-suite bootstrap.

The project keeps its importable code under ``src/`` and runs entry points with
``src`` on ``sys.path``; the tests mirror that so ``vio`` resolves the same way
in tests and in production.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
