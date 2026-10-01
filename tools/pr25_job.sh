#!/bin/bash
# PR #25 (transformer forward serial stages over host tasks): ab shas main vs branch on this box (fixture inputs),
# GPU transformer checks, CPU cells transformer-infer and samba-infer main vs branch.
set -uo pipefail
cd "$(dirname "$0")/.."
V=$1; O=/root/pr25-$V; rm -rf $O; mkdir -p $O; BR=$PWD; MAIN=/root/mojolearn-pr25main
if [ $V = nvidia ]; then AR=sm_89; else AR=gfx942; fi
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
(cd /root/mojolearn && git fetch -q origin main && rm -rf $MAIN && git worktree prune && git worktree add -f $MAIN FETCH_HEAD >/dev/null 2>&1)
cp $BR/tools/host_threads_ab_check.py $BR/tools/transformer_ab_input_sha.py $MAIN/tools/
TV=/root/torchvenv; [ -x $TV/bin/python ] || { $(cd $BR && pixi run python3 -c 'import sys;print(sys.executable)' | tail -1) -m venv --system-site-packages $TV; $TV/bin/python -m ensurepip -q >/dev/null 2>&1; $TV/bin/python -m pip -q install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1; }
for T in main branch; do D=$BR; [ $T = main ] && D=$MAIN; cd $D; pixi install > $O/pixi-$T.log 2>&1
  for b in neural_host mamba_host byte_lm_host; do MOJOLEARN_TARGET_COLUMN=cpu bash bindings/build_$b.sh > $O/build-$T-$b.log 2>&1; rc build-$T-$b $?; done
  for m in transformer mamba3; do for L in 512 2048; do MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_VENDOR=cpu PYTHONPATH=python timeout 1800 pixi run python tools/host_threads_ab_check.py --model $m --length $L --timing > $O/ab-$T-$m-$L.log 2>&1; rc ab-$T-$m-$L $?
    echo "$T $m $L $(grep -E "sha256" $O/ab-$T-$m-$L.log | grep -oE "one [0-9.]+ ms, policy [0-9.]+ ms|sha256 one [0-9a-f]+ policy [0-9a-f]+")" >> $O/shas.txt; done; done
  for l in transformer-infer samba-infer; do MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_VENDOR=cpu MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$PWD/python timeout 3600 $TV/bin/python tools/bench_board_neural.py race --lane $l --shape full --arms ours --rounds 5 --out $O/cell-$T-$l --work $O/work --ours-python $TV/bin/python > $O/cell-$T-$l.log 2>&1; rc cell-$T-$l $?
    echo "$T $l $(grep -oE "median_ms=[0-9.]+|mean_nll.{0,22}" $O/cell-$T-$l.log | head -2 | tr "\n" " ")" >> $O/cells.txt; done
done
cd $BR; export MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR
timeout 3600 pixi run check-transformer > $O/gpu-check-transformer.log 2>&1; rc gpu-check-transformer $?
timeout 3600 pixi run check-transformer-options > $O/gpu-check-transformer-options.log 2>&1; rc gpu-check-transformer-options $?
echo done > $O/done
