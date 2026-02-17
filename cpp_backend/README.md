# Agar Clone C++ Backend

This is a C++ port of the Python authoritative backend. It preserves the game simulation, bot logic, and websocket protocol used by the existing browser client.

## Build

Requires:

- C++20 compiler
- Boost (system, thread, json)

```bash
cmake -S cpp_backend -B cpp_backend/build
cmake --build cpp_backend/build
```

## Run

```bash
./cpp_backend/build/agario_server --host 0.0.0.0 --port 8099 --static-dir static
```

Open: `http://localhost:8099`

## Bots

You can set bot count or full specs via CLI:

```bash
./cpp_backend/build/agario_server --bots 24
./cpp_backend/build/agario_server --bot-specs "solo_smart:12,team_swarm:8:alpha"
./cpp_backend/build/agario_server --bots 128 --bot-threads 8
```
