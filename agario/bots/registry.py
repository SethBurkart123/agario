"""Bot plugin registry and dynamic loader."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence

from .types import BotBrain, BotInitContext

BotFactory = Callable[[BotInitContext], BotBrain]


class BotRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, BotFactory] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories.keys()))

    def register(self, name: str, factory: BotFactory) -> None:
        key = (name or "").strip().lower()
        if not key:
            raise ValueError("Bot plugin name cannot be empty")
        if key in self._factories:
            raise ValueError(f"Duplicate bot plugin registration: {key}")
        self._factories[key] = factory

    def create(self, name: str, init_ctx: BotInitContext) -> BotBrain:
        key = name.strip().lower()
        factory = self._factories.get(key)
        if factory is None:
            available = ", ".join(self.names) or "<none>"
            raise ValueError(f"Unknown bot plugin '{key}'. Available: {available}")
        return factory(init_ctx)


def load_plugin_modules(module_names: Sequence[str], registry: BotRegistry) -> None:
    for module_name in module_names:
        name = module_name.strip()
        if not name:
            continue
        module = importlib.import_module(name)
        register_fn = getattr(module, "register", None)
        if not callable(register_fn):
            raise ValueError(f"Bot plugin module '{name}' must define register(registry)")
        register_fn(registry)

