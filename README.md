# Agar Clone

An Agar.io-style simulator whose live server, bots, and authoritative physics
all run in one native Rust process.

```text
Axum HTTP/WebSocket tasks ──commands──▶ fixed 25 TPS simulation thread
                         ◀─latest frame─          │
                                          Rayon bot workers
```

The simulation thread exclusively owns the world. Network tasks cannot lock or
delay it, slow clients receive the newest frame instead of building a backlog,
and the built-in bot policies never cross a Python FFI boundary.

## Run

From the repository root:

```bash
AGARIO_BOT_SPECS="solo_smart:48,solo_smart_v2:1024" \
  cargo run --release --manifest-path agario_core/Cargo.toml
```

Open <http://localhost:8099>. The overview is at
<http://localhost:8099/overview>, and native bot status is exposed at
<http://localhost:8099/api/bots>.

The default population is `solo_smart:16`. Supported native policies are
`solo_smart` and `solo_smart_v2`; specs use
`policy[:count[:team[:name-prefix]]]`.

Useful settings:

```bash
AGARIO_PORT=8099
AGARIO_HOST=0.0.0.0
AGARIO_BOT_RANDOM_SEED=1337
RAYON_NUM_THREADS=16
```

## Controls

- Move: mouse
- Split: `Space`
- Eject mass: `W`

## Source layout

```text
agario_core/src/
├── main.rs                 # native executable
├── server.rs               # Axum, WebSockets, engine thread
├── world.rs                # authoritative mechanics
└── world/
    ├── native_bots.rs      # parallel built-in policies
    └── native_server.rs    # culled wire snapshots
static/                     # browser canvas client
bot_solutions/rl_v1/        # separate offline RL training solution
```

All live gameplay constants and formulas are in `agario_core/src/config.rs`
and `agario_core/src/world.rs`. The browser only interpolates and renders
server snapshots.

## Verification

```bash
cargo test --release --manifest-path agario_core/Cargo.toml
cargo check --release --manifest-path agario_core/Cargo.toml
```

The server reports achieved TPS, average busy time per tick, bot count, and
connected clients every five seconds. This makes overload visible instead of
silently slowing game time.

## Offline RL training

`bot_solutions/rl_v1` is an independent training workspace. Its Python tools
call the Rust batched arena engine for model training; Python is not loaded by
the live server.

```bash
uv run python -m bot_solutions.rl_v1.ppo --updates 600 --arenas 4
```
