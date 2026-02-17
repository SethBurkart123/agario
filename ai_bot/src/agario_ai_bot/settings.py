"""Configuration helpers for the tiny AlphaZero-style bot."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class AiBotSettings:
    model_path: str = "ai_bot/models/policy_value.pt"
    device_preference: str = "mps"
    planner_device_preference: str = "cpu"
    simulations: int = 48
    horizon_steps: int = 6
    discount: float = 0.96
    c_puct: float = 1.4
    default_target_distance: float = 1800.0
    direction_bins: int = 16
    magnitude_bins: tuple[float, ...] = (0.35, 0.65, 1.0)
    dirichlet_alpha: float = 0.35
    dirichlet_epsilon: float = 0.2
    train_decision_temperature: float = 1.0
    runtime_decision_interval: float = 1.0 / 12.0
    max_considered_actions: int = 24
    use_gpu_rollout: bool = True
    gpu_rollout_food_limit: int = 512
    gpu_rollout_ejected_limit: int = 128

    @classmethod
    def from_env(cls) -> AiBotSettings:
        defaults = cls()
        return cls(
            model_path=_env_str("AGARIO_AI_BOT_MODEL_PATH", defaults.model_path),
            device_preference=_env_str("AGARIO_AI_BOT_DEVICE", defaults.device_preference),
            planner_device_preference=_env_str("AGARIO_AI_BOT_PLANNER_DEVICE", defaults.planner_device_preference),
            simulations=max(4, _env_int("AGARIO_AI_BOT_SIMULATIONS", defaults.simulations)),
            horizon_steps=max(1, _env_int("AGARIO_AI_BOT_HORIZON_STEPS", defaults.horizon_steps)),
            discount=min(0.999, max(0.5, _env_float("AGARIO_AI_BOT_DISCOUNT", defaults.discount))),
            c_puct=max(0.2, _env_float("AGARIO_AI_BOT_C_PUCT", defaults.c_puct)),
            default_target_distance=max(
                200.0,
                _env_float("AGARIO_AI_BOT_TARGET_DISTANCE", defaults.default_target_distance),
            ),
            direction_bins=max(4, _env_int("AGARIO_AI_BOT_DIRECTION_BINS", defaults.direction_bins)),
            magnitude_bins=defaults.magnitude_bins,
            dirichlet_alpha=max(0.05, _env_float("AGARIO_AI_BOT_DIR_ALPHA", defaults.dirichlet_alpha)),
            dirichlet_epsilon=min(
                0.95,
                max(0.0, _env_float("AGARIO_AI_BOT_DIR_EPS", defaults.dirichlet_epsilon)),
            ),
            train_decision_temperature=max(
                0.05,
                _env_float("AGARIO_AI_BOT_TRAIN_TEMP", defaults.train_decision_temperature),
            ),
            runtime_decision_interval=max(
                1.0 / 40.0,
                _env_float("AGARIO_AI_BOT_RUNTIME_DECISION_INTERVAL", defaults.runtime_decision_interval),
            ),
            max_considered_actions=max(
                4,
                _env_int("AGARIO_AI_BOT_MAX_CONSIDERED_ACTIONS", defaults.max_considered_actions),
            ),
            use_gpu_rollout=_env_bool("AGARIO_AI_BOT_USE_GPU_ROLLOUT", defaults.use_gpu_rollout),
            gpu_rollout_food_limit=max(
                32,
                _env_int("AGARIO_AI_BOT_GPU_ROLLOUT_FOOD_LIMIT", defaults.gpu_rollout_food_limit),
            ),
            gpu_rollout_ejected_limit=max(
                16,
                _env_int("AGARIO_AI_BOT_GPU_ROLLOUT_EJECTED_LIMIT", defaults.gpu_rollout_ejected_limit),
            ),
        )
