#!/bin/bash
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
