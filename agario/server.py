"""FastAPI app and multiplayer websocket orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .bots.manager import BotManager
from .world import GameWorld


class RealtimeServer:
    def __init__(self) -> None:
        self.world = GameWorld()
        self.bot_manager = BotManager.from_config(self.world)
        self.connections: dict[str, WebSocket] = {}
        self.lock = asyncio.Lock()
        self._tick_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._tick_task is None:
            async with self.lock:
                self.bot_manager.ensure_started(now=time.perf_counter())
            self._tick_task = asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        if self._tick_task is not None:
            self._tick_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._tick_task
            self._tick_task = None

    async def connect(self, websocket: WebSocket, player_name: str) -> dict:
        now = time.perf_counter()
        async with self.lock:
            player = self.world.add_player(player_name=player_name, now=now)
            self.connections[player.id] = websocket

        return {
            "type": "welcome",
            "playerId": player.id,
            "name": player.name,
            "tickRate": config.TICK_RATE,
            "inputHz": config.INPUT_HZ,
            "world": {"w": config.WORLD_WIDTH, "h": config.WORLD_HEIGHT},
        }

    async def disconnect(self, player_id: str) -> None:
        async with self.lock:
            self.connections.pop(player_id, None)
            self.world.remove_player(player_id)

    async def handle_input(self, player_id: str, payload: dict) -> None:
        target = payload.get("target") or {}
        tx = target.get("x")
        ty = target.get("y")
        split = bool(payload.get("split", False))
        eject = bool(payload.get("eject", False))

        async with self.lock:
            self.world.set_input(player_id, tx, ty, split=split, eject=eject)

    async def _tick_loop(self) -> None:
        interval = 1.0 / config.TICK_RATE
        last = time.perf_counter()

        while True:
            tick_start = time.perf_counter()
            dt = min(0.1, tick_start - last)
            last = tick_start

            async with self.lock:
                self.bot_manager.tick(dt=dt, now=tick_start)
                self.world.update(dt=dt, now=tick_start)
                snapshots: list[tuple[str, WebSocket, dict]] = []
                for player_id, websocket in self.connections.items():
                    snapshot = self.world.snapshot_for(player_id)
                    if snapshot is None:
                        continue
                    snapshots.append((player_id, websocket, snapshot))

            if snapshots:
                results = await asyncio.gather(
                    *(self._safe_send_json(websocket, payload) for _, websocket, payload in snapshots),
                    return_exceptions=True,
                )
                dropped = [
                    player_id
                    for (player_id, _, _), result in zip(snapshots, results)
                    if result is False or isinstance(result, Exception)
                ]
                for player_id in dropped:
                    await self.disconnect(player_id)

            elapsed = time.perf_counter() - tick_start
            await asyncio.sleep(max(0.0, interval - elapsed))

    async def _safe_send_json(self, websocket: WebSocket, payload: dict) -> bool:
        try:
            await websocket.send_json(payload)
            return True
        except Exception:
            return False

app = FastAPI(title="Agar Clone")
state = RealtimeServer()

static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
async def startup_event() -> None:
    await state.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await state.stop()


@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/bots")
async def bot_status() -> dict:
    return state.bot_manager.describe()


@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket) -> None:
    await websocket.accept()
    player_id: str | None = None

    try:
        first = await websocket.receive_json()
        if first.get("type") != "join":
            await websocket.close(code=1003, reason="First message must be join")
            return

        name = str(first.get("name") or "Cell")
        welcome = await state.connect(websocket, name)
        player_id = welcome["playerId"]
        await websocket.send_json(welcome)

        while True:
            msg = await websocket.receive_json()
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong", "ts": msg.get("ts")})
                continue
            if msg.get("type") != "input" or player_id is None:
                continue
            await state.handle_input(player_id, msg)

    except WebSocketDisconnect:
        pass
    finally:
        if player_id is not None:
            await state.disconnect(player_id)
