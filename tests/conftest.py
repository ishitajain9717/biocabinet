"""Ensure the repository root is on sys.path when running pytest.

CI and local runs use `from scripts...` imports. That requires either:
  pip install -e .
or this conftest adding the project root to sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
