#!/bin/bash
# PR #58 (dq backward query rows per block) on one GPU box: released 0.8.33 + this tree's Python and
# neural bindings; lm-forward + lm-train-step for byte_lm built at TQ64 (default), TQ32, TQ16 (x2 each),
# then an ATTN_PHASE_TIMERS build of each arm for the dq phase tick.
set -uo pipefail
cd "$(dirname "$0")/.."
V=$1; O=/root/pr58-$V; rm -rf $O; mkdir -p $O
if [ $V = nvidia ]; then BK=cuda; AR=sm_89; PLUG=mojolearn-nvidia; else BK=hip; AR=gfx942; PLUG=mojolearn-amd; fi
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1 MOJOLEARN_COMPILE_JOBS=4 MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR MOJOLEARN_BENCH_INSTALLED=1; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
git rev-parse HEAD > $O/head.txt; pixi install > $O/pixi.log 2>&1
so_of() { ls -t python/mojolearn/identical/_mojolearn_$1.so python/mojolearn/_mojolearn_$1.so 2>/dev/null | head -1; }
bld() { rm -f python/mojolearn/identical/_mojolearn_$1.so python/mojolearn/_mojolearn_$1.so; MOJOLEARN_MOJO_BUILD_FLAGS="${3:-}" bash bindings/build_$1.sh > $O/build-$2.log 2>&1; rc build-$2 $?; s=$(so_of $1); [ -n "$s" ] && cp $s $O/$2.so; }
for m in transformer byte_lm training mamba linalg; do bld $m $m; done
bld byte_lm tq32 "-D MOJOLEARN_ATTN_DQ_TQ32=1"; bld byte_lm tq16 "-D MOJOLEARN_ATTN_DQ_TQ16=1"
bld byte_lm timers-tq64 "-D MOJOLEARN_ATTN_PHASE_TIMERS=1"; bld byte_lm timers-tq32 "-D MOJOLEARN_ATTN_PHASE_TIMERS=1 -D MOJOLEARN_ATTN_DQ_TQ32=1"; bld byte_lm timers-tq16 "-D MOJOLEARN_ATTN_PHASE_TIMERS=1 -D MOJOLEARN_ATTN_DQ_TQ16=1"
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1); $base -m venv $O/venv; P=$O/venv/bin/python
$P -m pip -q install mojolearn==0.8.33 $PLUG==0.8.33 numpy==2.5.2 scipy==1.18.0 > $O/pip.log 2>&1; $P -m pip -q install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu >> $O/pip.log 2>&1
S=$($P -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
T=$(dirname $(find $S/.. -path "*$BK/$AR/identical/_mojolearn_byte_lm.so" | head -1)); echo "$T" > $O/target.txt
cp $S/_version.py /tmp/v58.py; cp python/mojolearn/*.py $S/; cp /tmp/v58.py $S/_version.py; find $S -name __pycache__ -exec rm -rf {} +
inst() { cp $O/$1.so $T/_mojolearn_$2.so; }
for m in transformer byte_lm training mamba linalg; do inst $m $m; done
st() { env $2 timeout 3600 $P tools/neural_stage_timing.py --lane lm-forward --lane lm-train-step --calls 8 > $O/stage-$1.log 2>&1; rc stage-$1 $?; }
for i in 1 2; do for a in byte_lm tq32 tq16; do inst $a byte_lm; st $a-$i X=1; done; done
for a in tq64 tq32 tq16; do inst timers-$a byte_lm; st timers-$a X=1; done
inst byte_lm byte_lm
echo done > $O/done
