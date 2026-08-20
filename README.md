# Agar Clone (Python Multiplayer)

A modular Agar.io-style multiplayer prototype with an authoritative Python backend and browser canvas frontend.

## Features

- Real-time multiplayer over WebSockets
- Server-authoritative simulation loop
- Pluggable server-side bot framework (team-capable)
- Grid arena with world bounds
- Food spawning + growth
- Blob eating and leaderboard
- Split (`Space`)
- Eject mass (`W`)
- Virus hazards that split large blobs ("blow-up" behavior)
- Visibility culling + spatial hash broad-phase for better scalability

## Project Structure

```text
agario/
├── agario/
│   ├── config.py      # Tunables and constants
│   ├── bots/          # Bot plugin contracts + runtime manager
│   ├── models.py      # Dataclasses for entities
│   ├── spatial.py     # Spatial hash helper
│   ├── world.py       # Authoritative simulation
│   └── server.py      # FastAPI + websocket orchestration
├── bot_solutions/
│   ├── programmatic/  # Hand-written bot strategies
│   └── rl_v1/         # Neural bot, training stack, tools, and checkpoints
├── static/
│   ├── index.html
│   ├── styles.css
│   └── client.js
├── main.py            # Local run entrypoint
└── pyproject.toml
```

## Run

```bash
python -m pip install -e .
python main.py
```

Open: `http://localhost:8000`

## Controls

- Move: Mouse
- Split: `Space`
- Eject Mass: `W`

## Simulation Engines

Two interchangeable engines run the same physics:

- **Rust core** (`agario_core/`, default) — a parity-tested PyO3 port of the
  Python world. ~1,100x realtime on the full map vs ~15x for Python. Used by
  the live server and by RL training.
- **Python reference** (`agario/world.py`) — the readable source of truth.

Select with `AGARIO_ENGINE=rust|python`. Parity is enforced by
`uv run python -m tools.parity_check`, which drives both engines with an
identical seed + scripted inputs and compares full state tick-by-tick (the
Rust core embeds a CPython-compatible MT19937 so seeded runs match bitwise).
Benchmark either engine with `uv run python -m tools.bench_sim --engine rust`.

If you change game mechanics, change `agario/world.py` AND the mirrored code
in `agario_core/src/world.rs`, then run the parity check.

## RL Training (neural bots)

`bot_solutions/rl_v1/` contains the neural bot and its training stack over the
Rust core:

```bash
# train (outputs stay inside bot_solutions/rl_v1/checkpoints/)
uv run python -m bot_solutions.rl_v1.ppo --updates 600 --arenas 4

# watch metrics live at http://localhost:8123
uv run python -m bot_solutions.rl_v1.tools.train_dashboard

# evaluate vs the solo_smart heuristic bots
uv run python -m bot_solutions.rl_v1.eval

# play against the trained bots
AGARIO_BOT_PLUGIN_MODULES=bot_solutions.programmatic,bot_solutions.rl_v1.plugin \
AGARIO_BOT_SPECS=neural:8 uv run python main.py
```

The neural bot has difficulty knobs: `AGARIO_NEURAL_TEMPERATURE` (higher =
sloppier) and `AGARIO_NEURAL_THINK_SECONDS` (reaction delay). `--wandb` on the
trainer logs to Weights & Biases if installed.

## Notes for Extending

- Add new game mechanics in `agario/world.py` so simulation remains server-authoritative.
- Keep protocol changes coordinated between `agario/server.py` and `static/client.js`.
- Gameplay constants live in `agario/config.py` for quick balancing.

## Bot Plugin System

Bots are fully server-driven and use the same authoritative input path as human players.

- Plugin modules are configured by `AGARIO_BOT_PLUGIN_MODULES` (comma-separated python modules).
- Bot population is configured by `AGARIO_BOT_SPECS`.
- Dynamic scaling on bot elimination is controlled by:
  `AGARIO_BOT_SPAWN_ON_EATEN`, `AGARIO_BOT_SPAWN_PER_ELIMINATION`, `AGARIO_BOT_MAX_ACTIVE`.
- Runtime state is visible at `GET /api/bots`.

### Bot spec format

`plugin_name[:count[:team_id[:name_prefix]]]`

Examples:

- `solo_smart:16`
- `team_swarm:8:red:Red`
- `predator:4`
- `forager:6`

Combined:

```bash
AGARIO_BOT_SPECS="solo_smart:16"
```

### Dynamic bot scaling

By default, when a bot is eliminated, new bots are spawned (up to a cap) so matches can ramp in difficulty.

```bash
AGARIO_BOT_SPAWN_ON_EATEN=true
AGARIO_BOT_SPAWN_PER_ELIMINATION=1
AGARIO_BOT_MAX_ACTIVE=40
```

### Create a custom plugin

Create a module with a `register(registry)` function:

```python
from agario.bots.types import BotAction

class MyBrain:
    def decide(self, ctx):
        me = ctx.me.blobs[0] if ctx.me.blobs else None
        if me is None:
            return BotAction(ctx.world_width / 2, ctx.world_height / 2)
        return BotAction(me.x + 200, me.y)

def register(registry):
    registry.register("my_bot", lambda init_ctx: MyBrain())
```

Then include that module in:

```bash
AGARIO_BOT_PLUGIN_MODULES="bot_solutions.programmatic,my_project.my_bots"
AGARIO_BOT_SPECS="my_bot:10:alpha:Alpha"
```
