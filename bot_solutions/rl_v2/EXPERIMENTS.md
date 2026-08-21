# Experiment ledger

Append every meaningful attempt. Do not rewrite a failed result into a success;
later work may revisit it with a specific reason.

## R000 — RL V1 postmortem

**Status:** historical failure, useful components retained.

**Tried:** behavior cloning from `solo_smart`, symmetric PPO self-play, then
short-horizon Expert Iteration / AlphaZero-style search.

**Observed:** cloning remained below its noisy teacher; symmetric self-play
made dynamic actions unprofitable and split use collapsed from 8.4% to 0.14%;
the search teacher eventually scored about 628 mass against the raw policy's
1137, so further distillation became anti-learning.

**Why:** there was no stable promotion ladder, the reward favored safe
symmetric equilibria, and search was trusted after it became weaker than its
student.

**Keep:** Rust observations, batched arenas, recurrent inference, cloned-world
search, and the rule that a teacher must beat its student before distillation.

**Change:** establish paired evaluation and behavioral gates before training
another model.

## R001 — Native ladder baseline

**Status:** complete; candidate rejected.

**Hypothesis:** paired fixed-seed matches with insertion-order swaps and
time-averaged player utility will distinguish real policy improvements from
spawn luck and one late snowball.

**Candidate:** `solo_smart_v2` against `solo_smart`.

**Result:** six 180-second games over three paired seeds, 16 players per
policy. V2 scored **46.0%** and rated 995.2 against V1 at 1004.8. V2 achieved
higher mean mass (111.4 vs 102.7) and peak mass (238.8 vs 201.0), but fewer
kills (0.55 vs 0.74), far more deaths (0.24 vs 0.02), and 5.2 times as many
splits (3.56 vs 0.69). The runner sustained 226-282 simulated seconds per wall
second.

**Why:** V2's aggressive collection and group attacks grow quickly, but its
split payoff ignores enough of the vulnerable post-split period. Raw mass made
it look improved while paired utility exposed the survival and conversion
failure.

**Decision:** keep V1 as the initial ladder leader. The next candidate must
retain V2's growth while reducing unnecessary splits and post-split deaths.

## R002 — H1 risk memory

**Status:** promoted as the first RL V2 champion.

**Tried:** `rl_v2_h1` retains V2's collection and cluster logic, runs at 16.7
Hz, treats the eight seconds after a split as a vulnerable state, waits five
seconds before splitting again, and rejects a split if a nearby enemy could eat
the resulting half.

**Result:** against V1 on three paired seeds, H1 scored **67.0%**. Every seed
pair remained above 64%. Mean mass was 124.5 vs 102.3, kills 0.19 vs 0.15, and
deaths 0.073 vs 0.052. Compared with the original V2 run, H1 reduced deaths by
about 70% while keeping aggressive growth. It entered the ladder at 1021 Elo.

**Why:** remembering the post-split commitment prevents the rapid follow-up
splits and exposed-half attacks that made V2 look productive but lose games.

**Next:** isolate the benefit of reaction rate from the risk-memory change.

## R003 — Reaction-rate ablation

**Status:** useful ablation; not promoted.

**Tried:** `rl_v2_fast` is exactly V2's decisions at H1's 16.7 Hz, without the
new risk memory or split gate.

**Result:** on H1's same V1 seeds it scored only **51.5%**, versus H1's 67.0%.
In a direct H1 matchup on a second seed set, H1 scored 51.7%; it halved deaths
(0.083 vs 0.167), nearly doubled kills (0.68 vs 0.36), and split 30% less, but
the overall paired result was noisy.

**Why:** faster reactions recover much of V2's growth, but do not reproduce
H1's robust advantage against the stable V1 reference. The risk state changes
behavior in the intended direction, although more seeds are required before
claiming a decisive H1-vs-fast win.

**Decision:** keep H1 as champion; keep `rl_v2_fast` only as an ablation anchor.
