#!/bin/bash
# tools/gnm_mac_job.sh <builds,comma> <python script> [args...]   (lane/gap-neural-models, Apple boxes)
# Builds the named bindings (IDENTICAL, Metal) in this tree, then runs the
# script with this tree's python package under the board venv. Prints one
# GNM-BUILD line per build. Runs on the Mac queue only; never on the laptop.
set -u
builds=$1; shift
export MOJOLEARN_NUMERIC_MODE=${MOJOLEARN_NUMERIC_MODE:-identical} MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-4} PYTHONUNBUFFERED=1
for b in ${builds//,/ }; do
  bash bindings/build_$b.sh > /tmp/gnm-build-$b.log 2>&1; rc=$?
  echo "GNM-BUILD $b rc=$rc"
  [ $rc = 0 ] || grep -m 5 -A 5 'error' /tmp/gnm-build-$b.log
done
VP=
for d in board-0834 board-0833; do [ -x $HOME/$d/cache/venv/bin/python ] && { VP=$HOME/$d/cache/venv/bin/python; break; }; done
[ -n "$VP" ] || { echo "GNM no board venv"; exit 1; }
PYTHONPATH=$PWD/python "$VP" "$@"
