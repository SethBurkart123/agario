"""Launch the game server with ai_search bots from a chosen checkpoint."""

from __future__ import annotations

import argparse
import os

from ._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import torch
import uvicorn


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run agario server with ai_search bots."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument("--bots", type=int, default=16, help="Number of ai_search bots")
    parser.add_argument(
        "--simulations", type=int, default=32, help="PUCT simulations per decision"
    )
    parser.add_argument(
        "--decision-hz", type=float, default=12.0, help="Runtime decision frequency"
    )
    parser.add_argument(
        "--max-considered-actions",
        type=int,
        default=24,
        help="Top-K actions considered by search",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Force bot inference device (defaults to auto-detect)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = _build_parser().parse_args()
    if args.device is not None:
        os.environ["AGARIO_AI_BOT_DEVICE"] = args.device
    elif not os.getenv("AGARIO_AI_BOT_DEVICE"):
        os.environ["AGARIO_AI_BOT_DEVICE"] = _auto_device()
    os.environ["AGARIO_BOT_PLUGIN_MODULES"] = (
        "agario.bot_plugins.core,agario_ai_bot.runtime_plugin"
    )
    os.environ["AGARIO_BOT_SPECS"] = f"ai_search:{max(1, args.bots)}"
    os.environ["AGARIO_AI_BOT_MODEL_PATH"] = args.checkpoint
    os.environ["AGARIO_AI_BOT_SIMULATIONS"] = str(max(4, args.simulations))
    os.environ["AGARIO_AI_BOT_MAX_CONSIDERED_ACTIONS"] = str(
        max(4, args.max_considered_actions)
    )
    hz = max(1.0, args.decision_hz)
    os.environ["AGARIO_AI_BOT_RUNTIME_DECISION_INTERVAL"] = str(1.0 / hz)
    uvicorn.run("agario.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
