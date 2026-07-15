"""Ensure the repository root is on sys.path when running pytest.

CI and local runs use `from scripts...` imports. That requires either:
  pip install -e .
or this conftest adding the project root to sys.path.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Numba (used internally by scanpy) tries to cache JIT-compiled functions next
# to its source files. When scanpy is installed in a read-only location (e.g.
# /opt/homebrew or a CI container), that write fails with RuntimeError.
# Pointing NUMBA_CACHE_DIR at a writable temp directory fixes it without
# disabling JIT entirely.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / "numba_cache"),
)
