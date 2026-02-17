"""Runtime bootstrap helpers for local workspace imports."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path() -> None:
    # ai_bot/src/agario_ai_bot/_bootstrap.py -> repo root at parents[3]
    root = Path(__file__).resolve().parents[3]
    if not (root / "agario" / "__init__.py").exists():
        return
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

