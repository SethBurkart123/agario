# Benchmark contract

## Match protocol

- Equal policy populations share one authoritative Rust arena.
- Every seed is played twice with policy insertion order reversed.
- The simulation runs uncapped and samples each player once per simulated
  second.
- A player utility is based on time-averaged square-root mass, kills, and
  deaths. Square-root mass limits the influence of a single runaway giant.
- A policy's match score is the fraction of cross-policy player comparisons it
  wins. Elo consumes this continuous score rather than a binary winner.

## Recorded metrics

- time-averaged mass and peak mass;
- kills and deaths per player;
- average owned-cell count;
- successful split and eject events;
- paired match score and Elo update;
- simulated seconds per wall-clock second.

## Promotion gate

A candidate becomes champion only after at least three paired seeds and when:

1. its mean paired score is above 0.5;
2. the 95% bootstrap interval does not show a material loss;
3. deaths do not increase by more than 15% without compensating growth;
4. early-game mass and tactical-suite performance do not regress by more than
   5%; and
5. inference meets the 15 Hz population budget.

The initial ladder implementation supplies the fixed-seed FFA layer. Tactical,
social, and learned-policy runners are added as experiments demand them rather
than being invented ahead of evidence.
