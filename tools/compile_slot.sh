#!/bin/bash
# Owner 2026-10-22: dedicated Apple fast build work is stopped.
# Refuse before taking a slot; existing controllers can continue to identical.
# 2026-10-23 (Andrew: "that may not be a rule we want to follow now"): integration and release builds may compile
# the FAST set (MOJOLEARN_INTEG_BUILD=1); dedicated Apple FAST lane work stays paused.
if [ "${MOJOLEARN_NUMERIC_MODE:-}" = fast ] && [ "${MOJOLEARN_INTEG_BUILD:-0}" != 1 ]; then
  echo "REFUSED: Apple fast builds are paused by owner instruction 2026-10-22 (integration builds set MOJOLEARN_INTEG_BUILD=1); continue identical." >&2
  exit 78
fi
# compile_slot.sh <command...>: (versioned copy; origin ~/mojolearn-evidence/compile_slot.sh, now a shim that execs this file)
# compile_slot.sh <command...>: run a compile holding one of 4 machine-wide slots (mkdir locks).
# Every lane subagent wraps its mojo builds in this so at most 4 compile at once on the M4.
S=$HOME/mojolearn-evidence/compile-slots; mkdir -p $S
while :; do
  for i in 1 2 3 4; do
    if mkdir $S/slot$i 2>/dev/null; then
      echo $$ > $S/slot$i/pid
      trap "rm -rf $S/slot$i" EXIT INT TERM
      nice -n 19 "$@"; exit $?
    fi
    # reclaim a slot whose holder died
    p=$(cat $S/slot$i/pid 2>/dev/null); [ -n "$p" ] && ! kill -0 $p 2>/dev/null && rm -rf $S/slot$i
  done
  sleep 15
done
