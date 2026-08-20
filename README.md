# Agar Clone

A multiplayer Agar.io-style simulator with an authoritative Rust core, Python server orchestration, and a browser canvas client.

## Features

- Real-time multiplayer over WebSockets
- Server-authoritative simulation loop
- Pluggable server-side bot framework (team-capable)
- Grid arena with world bounds
- Food spawning + growth
- Blob eating and leaderboard
- Split (`Space`)
- Eject mass (`W`)
- Feedable viruses that grow, launch children, and pop player cells
- Visibility culling + spatial hash broad-phase for better scalability

## Project Structure

```text
agario/
├── agario/
│   ├── config.py      # Server and bot-process settings
│   ├── bots/          # Bot plugin contracts + runtime manager
│   └── server.py      # FastAPI + websocket orchestration
├── agario_core/       # Authoritative Rust simulation and RL batch engine
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

## Simulation

`agario_core/` is the only simulation implementation. It is used by the live
server, programmatic bots, and RL training. Gameplay defaults and formulas live
in `agario_core/src/config.rs` and `agario_core/src/world.rs`; Python reads the
small public settings map exposed by the extension.

Benchmark with `uv run python -m tools.bench_sim`.

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

- Add game mechanics in `agario_core/` so every consumer uses the same rules.
- Keep protocol changes coordinated between `agario/server.py` and `static/client.js`.
- Keep `agario/config.py` limited to server and bot-process settings.

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

Use the stateful second-generation solo bot on its own or beside the original:

```bash
AGARIO_BOT_SPECS="solo_smart:8,solo_smart_v2:8" uv run python main.py
uv run python -m tools.battle_bots --seconds 180 --seeds 1337 2027 4099
```

`solo_smart_v2` cycles through guardian, hunter, trickster, and grazer trait
profiles. Add `--start-mass 400` to the battle command to stress split combat,
virus play, and fragmented recovery immediately.

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
