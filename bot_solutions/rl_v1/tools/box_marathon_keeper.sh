#!/bin/bash
# Keep the AlphaZero marathon running indefinitely: when a budget expires (or
# a crash kills it), relaunch — state.json resumes mid-cycle, so nothing is
# lost. Stop cleanly with:
#   touch ~/agario/bot_solutions/rl_v1/checkpoints/marathon/STOP
# Circuit-breaker: 4 exits in under 10 minutes each = something is broken;
# give up loudly instead of hot-looping.
set -u
cd "$HOME/agario"
CHECKPOINTS=bot_solutions/rl_v1/checkpoints
STOP=$CHECKPOINTS/marathon/STOP
rm -f "$STOP"
fails=0

echo "keeper: started $(date)" >> /tmp/azero_marathon.log

while true; do
  if [ -f "$STOP" ]; then
    echo "keeper: STOP file found, exiting $(date)" >> /tmp/azero_marathon.log
    exit 0
  fi
  if pgrep -f "[b]ot_solutions.rl_v1.marathon" > /dev/null; then
    sleep 60
    continue
  fi
  echo "keeper: (re)launching marathon $(date)" >> /tmp/azero_marathon.log
  rm -f "$CHECKPOINTS/marathon/marathon.pid"
  start=$(date +%s)
  .venv/bin/python -W ignore -m bot_solutions.rl_v1.marathon --hours 8 \
    >> /tmp/azero_marathon.log 2>&1
  dur=$(( $(date +%s) - start ))
  if [ "$dur" -lt 600 ]; then
    fails=$((fails + 1))
  else
    fails=0
  fi
  if [ "$fails" -ge 4 ]; then
    echo "keeper: marathon crashing repeatedly (4x under 10min) — giving up $(date)" \
      >> /tmp/azero_marathon.log
    exit 1
  fi
  sleep 30
done
