"""World engine selection: Rust core (agario_core) or pure-Python reference.

The Rust core is a parity-tested port of world.py (see tools/parity_check.py)
running ~250x faster. The Python world remains the reference implementation.
"""

from __future__ import annotations

import logging

from . import config
from .world import GameWorld

logger = logging.getLogger(__name__)


def create_world(seed: int | None = None):
    if config.ENGINE == "rust":
        try:
            from agario_core import CoreWorld

            return CoreWorld(seed)
        except ImportError:
            logger.warning(
                "AGARIO_ENGINE=rust but agario_core is not installed "
                "(run: uv pip install ./agario_core); falling back to Python engine"
            )
    return GameWorld(seed)
