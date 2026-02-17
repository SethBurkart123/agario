"""Model loading/saving helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import PolicyValueNet


def create_device(preferred: str) -> torch.device:
    choice = (preferred or "mps").strip().lower()
    if choice == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_or_init_model(
    *,
    model_path: str,
    obs_size: int,
    action_count: int,
    device: torch.device,
    hidden_size: int = 256,
) -> PolicyValueNet:
    model = PolicyValueNet(obs_size=obs_size, action_count=action_count, hidden_size=hidden_size)
    path = Path(model_path)
    if path.exists():
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def save_model(model: PolicyValueNet, model_path: str) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

