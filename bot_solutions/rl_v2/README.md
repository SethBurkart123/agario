# RL V2

RL V2 is the measured self-improvement track for an efficient Agar.io agent.
It does not replace the simulator or copy RL V1. It reuses the authoritative
Rust world and earns promotion through reproducible matches.

## Goal

Produce a compact recurrent policy that can run for hundreds of agents at
15 Hz or faster and develops useful long-horizon play: feeding, hunting,
escaping, virus use, restraint, and eventually cooperation.

The benchmark is the source of truth. A policy is better only when it wins
paired, fixed-seed matches without regressing badly on survival or basic
growth. Interesting clips are evidence for diagnosis, not promotion.

## Research loop

```text
scenario curriculum + mixed self-play
                 │
                 ▼
      candidate policy/checkpoint
                 │
       fixed seeds, side swaps
                 ▼
 tactical suite ── FFA ladder ── social league
                 │
         promote or reject
                 │
                 └──────▶ append EXPERIMENTS.md
```

1. Benchmark the current champion and candidate.
2. Find the largest measured weakness.
3. Make the smallest change that targets it.
4. Re-run the same benchmark and confidence gate.
5. Promote only a sustained improvement; record failures too.

## Agent contract

- **Rate:** at least 15 decisions per simulated second.
- **Input:** own-cell tokens, owner-aware nearby-cell tokens, viruses, and a
  polar density field for pellets/ejected mass. Raw pellet lists are avoided.
- **Memory:** a small recurrent state (initial target: 96-128 floats) carries
  intent, recent danger, partner reliability, and action history.
- **Action:** aim direction, target distance, split, and eject. Invalid or
  suicidal operations are masked by mechanics, not learned through thousands
  of needless deaths.
- **Runtime:** batched native inference. Training may use accelerator tooling,
  but the live server must not depend on Python.

The first learned model should stay below roughly 250k parameters. Capacity is
only increased after the benchmark demonstrates underfitting.
The concrete observation, network, curriculum, and social-learning design is
in [MODEL.md](MODEL.md).

## Evaluation

The ladder has four layers; a single Elo number is not allowed to hide a
catastrophic weakness.

1. **Mechanics:** pellet collection, escape, split range, virus avoidance.
2. **Tactics:** prey clusters, corner traps, split restraint, fragmented play.
3. **FFA:** mixed-policy, fixed-seed matches with insertion-order side swaps.
4. **Social:** paired rewards against selfish, cooperative, and exploitative
   opponents. Teaming is not expected to emerge from purely selfish symmetric
   FFA rewards.

Promotion requires a positive paired result and no major regression in deaths,
early growth, or invalid action rate. Ratings and behavioral metrics are stored
under `results/`; every attempt is explained in `EXPERIMENTS.md`.

## Commands

Run the native ladder benchmark:

```bash
cargo run --release --manifest-path agario_core/Cargo.toml \
  --bin rl_v2_ladder -- --games 6 --seconds 180
```

Watch the current live population beside the latest research results:

```bash
AGARIO_BOT_SPECS="rl_v2_h1:32,solo_smart:32" \
  cargo run --release --manifest-path agario_core/Cargo.toml
```

Then open <http://localhost:8099/lab>.
