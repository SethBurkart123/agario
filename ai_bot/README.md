## agario-ai-bot

Tiny AlphaZero-style bot package for this repo.

### What this includes

- Small shared policy/value network.
- PUCT-style rollout search (test-time compute) using cloned `GameWorld` states.
- Self-play trainer that learns from:
  - policy target: search visit distribution
  - value target: discounted returns
- Runtime plugin module: `agario_ai_bot.runtime_plugin`

### Train

From the workspace root:

```bash
uv run agario-ai-train --episodes 200 --players 8 --simulations 48 --device mps
```

High-throughput CUDA profile (search on CPU threads, learner on GPU):

```bash
uv run agario-ai-train \
  --episodes 200 \
  --players 8 \
  --simulations 24 \
  --max-considered-actions 12 \
  --device cuda \
  --planner-device cpu \
  --search-workers 8
```

GPU-bound rollout profile (training-search simulation on CUDA):

```bash
uv run agario-ai-train \
  --episodes 200 \
  --players 8 \
  --simulations 48 \
  --device cuda \
  --planner-device cuda \
  --gpu-rollout \
  --max-considered-actions 24
```

Model default path:

`ai_bot/models/policy_value.pt`

### Run in game

Use the plugin module and bot spec:

```bash
AGARIO_BOT_PLUGIN_MODULES="agario.bot_plugins.core,agario_ai_bot.runtime_plugin" \
AGARIO_BOT_SPECS="ai_search:16" \
uv run main.py
```

Run directly with a specific checkpoint:

```bash
uv run agario-ai-demo --checkpoint ai_bot/models/policy_value.pt --bots 16 --simulations 32 --decision-hz 12
```

Open full-map spectator mode at:

`http://localhost:8000/overview`

### Notes

- The model is intentionally small so one model instance can drive many bots.
- Search budget is controlled with env var `AGARIO_AI_BOT_SIMULATIONS`.
- Runtime decision frequency is controlled with `AGARIO_AI_BOT_RUNTIME_DECISION_INTERVAL` (seconds).
- Start with lower simulations for throughput and increase as needed.
