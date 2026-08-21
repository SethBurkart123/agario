# Compact policy design

## Perception

The policy should not receive a long unordered pellet list. It should receive
stable summaries whose cost does not grow with map population.

| Stream | Representation |
|---|---|
| Self | total/large/small mass, center velocity, cell count, merge timers, cooldowns |
| Own cells | up to 16 relative cell tokens with mass, velocity, boost, merge time |
| Other players | up to 24 **owner-level** tokens: total mass, largest cell, nearest cell, cell count, velocity, split threat/reward |
| Food | 12 angular × 4 radial bins containing mass, count, and nearest distance |
| Viruses | up to 8 relative tokens with size, fed state, velocity, and pop danger |
| Ejected mass | 12 bins split into own/other mass and flow direction |

Owner-level opponent tokens matter. RL V1's nearest 12 blob slots can spend the
whole budget on one fragmented player and lose the fact that a nearby group is
one owner, prey cluster, or coordinated threat.

All positions are egocentric and rotated by the agent's current heading. The
observation also includes the previous action and two seconds of event flags
(mass gained, cell lost, nearby split, feed received). This makes reciprocity
and opponent intent learnable without an ever-growing frame history.

## Network

```text
cell/player/virus tokens ─▶ shared 2-layer encoders ─▶ sum + max pooling
food/eject polar fields ─▶ small MLP ───────────────────────────────┐
self features ─▶ small MLP ────────────────────────────────────────┤
                                                                  ▼
                                                             GRU(128)
                                                     ┌────────┼────────┐
                                                     ▼        ▼        ▼
                                                   turn    distance  operation
```

DeepSets-style sum/max pooling is cheaper and easier to batch than attention,
while remaining permutation invariant. Start around 100k-200k parameters. The
GRU state is the bot's persistent intent and social memory.

The action heads are deliberately ordinary:

- 9 relative turns: straight, small/medium/hard left and right, reverse;
- 3 target distances: brake, steer, full reach;
- one operation: none, split, or eject.

Mechanics masks remove impossible split/eject actions. A short operation latch
prevents a noisy policy from turning one decision into accidental button spam.

## Learning sequence

1. **Behavior clone** H1 and tactical scenario oracles to learn movement,
   collection, and safe control without wasting RL samples.
2. **Recurrent league RL** against frozen historical policies, H1, selfish
   hunters, and deliberately exploitable bots. Pure mirror self-play is banned.
3. **Prioritized tactical scenarios** make rare split, virus, corner, and
   fragmented-cell decisions common enough to learn.
4. **Gated search** is used only around high-value tactical decisions. Search
   labels are accepted only when searched play beats the raw policy.
5. **Distill** useful league specialists back into the compact live policy.

## Making cooperation possible

Stable teaming is unlikely to appear from an entirely selfish reward: feeding
another agent is an immediate loss and symmetric self-play rewards defection.
The social league therefore mixes hidden episode incentives:

- selfish FFA agents;
- paired agents sharing part of their return;
- defectors that accept feeds but never reciprocate;
- loyal partners and opportunistic temporary partners.

The policy is not told a permanent team identity. It must infer reliability
from proximity, feeding, restraint, and joint attacks stored in the GRU. Social
evaluation records joint surplus, reciprocal feeding, time together, deaths
prevented, and exploitability by defectors. A cooperative specialist is only
promoted if it also remains viable when no trustworthy partner exists.
