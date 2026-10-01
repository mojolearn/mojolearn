#!/bin/bash
# PR #26 (Mamba-3 kscale estimate, inter-chunk state-row blocks): ab shas main vs branch (and PBLOCKS 8 / 1), cells.
set -uo pipefail
cd "$(dirname "$0")/.."
V=$1; O=/root/pr26-$V; rm -rf $O; mkdir -p $O; BR=$PWD; MAIN=/root/mojolearn-pr26main
until test -f /root/pr25-$V/done; do sleep 30; done
export PATH=/root/.pixi/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1 MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_VENDOR=cpu; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
(cd /root/mojolearn && git fetch -q origin main && rm -rf $MAIN && git worktree prune && git worktree add -f $MAIN FETCH_HEAD >/dev/null 2>&1)
cp $BR/tools/host_threads_ab_check.py $BR/tools/transformer_ab_input_sha.py $MAIN/tools/
TV=/root/torchvenv
for T in main branch; do D=$BR; [ $T = main ] && D=$MAIN; cd $D; pixi install > $O/pixi-$T.log 2>&1
  for b in neural_host mamba_host; do bash bindings/build_$b.sh > $O/build-$T-$b.log 2>&1; rc build-$T-$b $?; done
  for L in 512 2048; do for pb in default 8 1; do [ $T = main ] && [ $pb != default ] && continue
    if [ $pb = default ]; then unset MOJOLEARN_M3_INTER_PBLOCKS; else export MOJOLEARN_M3_INTER_PBLOCKS=$pb; fi
    PYTHONPATH=python timeout 1800 pixi run python tools/host_threads_ab_check.py --model mamba3 --length $L --timing > $O/ab-$T-$L-$pb.log 2>&1; rc ab-$T-$L-$pb $?
    echo "$T L$L pblocks=$pb $(grep -E "sha256" $O/ab-$T-$L-$pb.log | grep -oE "one [0-9.]+ ms, policy [0-9.]+ ms|sha256 one [0-9a-f]+ policy [0-9a-f]+" | tr "\n" " ") $(grep -E "m3.(kscale|inter_chunk) " $O/ab-$T-$L-$pb.log | tr -s " " | tr "\n" " ")" >> $O/shas.txt; done; done; unset MOJOLEARN_M3_INTER_PBLOCKS
  for l in mamba3-infer samba-infer; do MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$PWD/python timeout 3600 $TV/bin/python tools/bench_board_neural.py race --lane $l --shape full --arms ours --rounds 5 --out $O/cell-$T-$l --work $O/work --ours-python $TV/bin/python > $O/cell-$T-$l.log 2>&1; rc cell-$T-$l $?
    echo "$T $l $(grep -oE "median_ms=[0-9.]+|mean_nll.{0,22}" $O/cell-$T-$l.log | head -2 | tr "\n" " ")" >> $O/cells.txt; done
done
echo done > $O/done
