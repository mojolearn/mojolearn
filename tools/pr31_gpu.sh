#!/bin/bash
# PR #31 on a GPU box: training binding built from main and from the branch; optimizer_resident_check adam/sgd/adamw.
set -uo pipefail
cd "$(dirname "$0")/.."
V=$1; O=/root/pr31-$V; rm -rf $O; mkdir -p $O; BR=$PWD; MAIN=/root/mojolearn-pr31main
if [ $V = nvidia ]; then AR=sm_89; else AR=gfx942; fi
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1 MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR MOJOLEARN_COMPILE_JOBS=4; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
(cd /root/mojolearn && git fetch -q origin main && rm -rf $MAIN && git worktree prune && git worktree add -f $MAIN FETCH_HEAD >/dev/null 2>&1)
for T in main branch; do D=$BR; [ $T = main ] && D=$MAIN; cd $D; pixi install > $O/pixi-$T.log 2>&1
  for b in build build_training; do bash bindings/$b.sh > $O/$b-$T.log 2>&1; rc $b-$T $?; done
  for k in adam sgd adamw; do PYTHONPATH=python timeout 1800 pixi run python tools/optimizer_resident_check.py --kind $k > $O/opt-$T-$k.log 2>&1; rc opt-$T-$k $?
    echo "$T $k $(grep -E "resident moments|CHECK" $O/opt-$T-$k.log | tr -s " " | tr "\n" " ")" >> $O/opt.txt; done
done
echo done > $O/done
