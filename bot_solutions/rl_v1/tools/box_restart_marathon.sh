#!/bin/bash
# Restart the AlphaZero marathon on the GPU box with clean dashboard data.
# Self-match-safe kills; archives pre-azero history; truncates stale metrics.
set -u
cd "$HOME/agario"
CHECKPOINTS=bot_solutions/rl_v1/checkpoints

pkill -f "[b]ot_solutions.rl_v1.marathon" 2>/dev/null
pkill -f "[b]ot_solutions.rl_v1.azero" 2>/dev/null
pkill -f "[b]ot_solutions.rl_v1.imitate" 2>/dev/null
sleep 2
if pgrep -f "[b]ot_solutions.rl_v1" > /dev/null; then
  echo "ERROR: training processes still alive"; exit 1
fi
echo "training processes stopped"

rm -f "$CHECKPOINTS/marathon/marathon.pid"

python3 - <<'EOF'
import json
from pathlib import Path

h = Path("bot_solutions/rl_v1/checkpoints/marathon/history.jsonl")
rows = [json.loads(l) for l in h.read_text().splitlines() if l.strip()]
keep = [
    r for r in rows
    if r.get("type") == "gate"
    or "azero" in str(r.get("checkpoint", ""))
    or str(r.get("checkpoint", "")).endswith("latest.pt")
]
old = [r for r in rows if r not in keep]
Path("bot_solutions/rl_v1/checkpoints/marathon/history_pre_azero.jsonl").write_text(
    "".join(json.dumps(r) + "\n" for r in old)
)
h.write_text("".join(json.dumps(r) + "\n" for r in keep))
print(f"history: kept {len(keep)} azero-era rows, archived {len(old)}")
EOF

: > "$CHECKPOINTS/rl/metrics.jsonl"
echo "metrics.jsonl truncated"

setsid nohup .venv/bin/python -W ignore -m bot_solutions.rl_v1.marathon --hours 8 \
  >> /tmp/azero_marathon.log 2>&1 < /dev/null &
sleep 12
echo "--- status.json:"
cat "$CHECKPOINTS/marathon/status.json" 2>/dev/null || echo "(missing!)"
echo
echo "--- marathon:"
pgrep -f "[b]ot_solutions.rl_v1.marathon" > /dev/null && echo alive || echo DEAD
tail -1 /tmp/azero_marathon.log
