# Proper AlphaZero setup — implementation plan

Goal: a self-improvement loop whose training targets are *always stronger
than the current policy*. Everything below is grounded in measurements from
the 2026-06-11 session (see memory / git history).

## Why previous attempts plateaued
- Imitating solo_smart: caps at teacher level minus label noise (~80% turn
  acc floor from the expert's internal RNG). Measured.
- PPO self-play: optimizes vs its own twin; symmetric worlds make dynamic
  play unprofitable (split usage 8.4% -> 0.14% across phases). Measured.
- ExIt v1 search (10 candidates x 1 sample x 0.64s, non-dodging opponents):
  was stronger than the weak early policy (gen1-2 breakthroughs: 0% -> 50%
  wins) but is now WEAKER than the mature policy (head-to-head: search 628
  mass vs raw 1137). Distilling it = anti-learning. Measured.

## Component 1: Rust SearchPool (agario_core/src/search_pool.rs)
Thousands of cloned worlds stepping in lockstep, GIL-free, rayon-parallel.
API (Python-facing):
- `start(roots: Vec<(arena_idx, agent_id, k_candidates, m_samples)>,
   first_actions: flat)` — fast_clone the arena world per (candidate,
   sample); apply candidate first-actions; samples differ by forcing the
   FIRST opponent decision to vary (sample index seeds a small action
   perturbation) so averages cover branching futures.
- `observe_all() -> bytes` — obs for EVERY policy-driven player in every
  clone (focal agent + all opponents). One call.
- `apply_all(actions: flat)` + `tick_all()` — apply the batched GPU policy
  decisions, step frame_skip ticks (rayon over clones).
- `scores() -> Vec<f32>` — per-clone accumulated shaped reward (continuing
  death semantics, env.cfg constants) for the focal agent.
Python loop per decision: start -> [observe_all -> ONE batched GPU forward
-> apply_all -> tick_all] x horizon -> scores -> value-bootstrap final obs
(one more batched forward) -> average over samples -> per-candidate Q.
Budget: 24 arenas, 2 searched agents/arena/decision (round-robin), 16
candidates x 4 samples, horizon 12-16. ~3K clones, ~36K-row forwards.
This is the component that actually uses the 3080.

## Component 2: Soft targets
Per searched agent: tau-softmax over candidate Q -> marginal turn / op
distributions (candidates map to (turn, op) pairs; speed stays the policy's
continuous output, supervised as before). Chunk format adds `turn_soft
(T,B,8)`, `op_soft (T,B,3)`, with hard labels kept for metrics. Trainer:
soft cross-entropy when soft arrays present.

## Component 3: Value targets (BUILT — keep)
exit.py-style returns-to-go with truncation bootstrap + stored h0; trainer
MSEs the value head. Port into the new generator unchanged.

## Component 4: The gate (BUILT as inline script — formalize)
`azero.py gate`: 2 learners same arena, one search-driven one raw, >= 6
episodes. Require search mass > raw mass (and kills >=) before any
distillation. Run at the start of EVERY generation; abort generation if it
fails (then the fix is more search budget, never "distill anyway").

## Component 5: The loop (marathon shell already exists)
marathon stages: [gate -> generate(search) -> distill(policy soft-CE +
value MSE + speed MSE) -> eval(14ep, sem) -> ratchet]. No PPO inside the
loop. The 150M-step PufferLib run seeds the initial policy+value.

## Order of work (each step gated by its own smoke test)
1. SearchPool in Rust + python driver; benchmark clones/s and forward batch.
2. Gate script formalized (reuse session's inline version).
3. Soft-target generation + trainer support.
4. Wire into marathon; 2-generation pilot; verify gate passes and eval
   climbs above the 1089/71% champion.
5. Then scale model (D_MODEL 128, GRU 512) — capacity finally pays once
   targets exceed the student.

## Known constraints
- GPU box: cu128 torch pinned -> `uv run --no-sync`; pufferlib needs
  numpy<2 and source build (see memory).
- Keep disk < 85%.
- Eval noise: only trust deltas > 2 sem or sustained over 3+ generations.
