"""Bot plugin framework."""

from .manager import BotManager, parse_bot_specs
from .registry import BotRegistry, load_plugin_modules
from .types import BotAction, BotBrain, BotContext, BotInitContext, BotSpec

__all__ = [
    "BotAction",
    "BotBrain",
    "BotContext",
    "BotInitContext",
    "BotManager",
    "BotRegistry",
    "BotSpec",
    "load_plugin_modules",
    "parse_bot_specs",
]

