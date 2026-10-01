#!/bin/bash
# Main vs one or more neural branches on one GPU box (NVIDIA or AMD), no env knobs: each tree's neural bindings are
# built from source (plain and with -D MOJOLEARN_ATTN_PHASE_TIMERS=1), laid over the released 0.8.33 wheel in turn,
# and timed with tools/neural_stage_timing.py (lm-forward, lm-train-step), twice each, interleaved.
# Usage: tools/ab_job.sh <nvidia|amd> <tag> <branch>... (main is always the first arm). Out: /root/ab-<tag>-<vendor>.
set -uo pipefail
V=$1; TAG=$2; shift 2; O=/root/ab-$TAG-$V; rm -rf $O; mkdir -p $O
if [ $V = nvidia ]; then BK=cuda; AR=sm_89; PLUG=mojolearn-nvidia; else BK=hip; AR=gfx942; PLUG=mojolearn-amd; fi
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1 MOJOLEARN_COMPILE_JOBS=4 MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$AR MOJOLEARN_BENCH_INSTALLED=1; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
R=/root/ab-repo; [ -d $R/.git ] || git clone -q https://github.com/mojolearn/mojolearn.git $R
cd $R && git fetch -q origin
ARMS=main; for b in "$@"; do ARMS="$ARMS $b"; done
MODS="transformer byte_lm training"
for b in $ARMS; do a=$(echo $b | tr / _); t=$O/tree-$a
  git -C $R worktree prune; git -C $R worktree add -q -f --detach $t origin/$b; git -C $t rev-parse HEAD > $O/head-$a.txt
  cd $t; pixi install > $O/pixi-$a.log 2>&1
  for m in $MODS; do for k in plain timers; do
    f=""; [ $k = timers ] && f="-D MOJOLEARN_ATTN_PHASE_TIMERS=1"; [ $k = timers ] && [ $m != byte_lm ] && continue
    rm -f python/mojolearn/identical/_mojolearn_$m.so python/mojolearn/_mojolearn_$m.so
    MOJOLEARN_MOJO_BUILD_FLAGS="$f" bash bindings/build_$m.sh > $O/build-$a-$m-$k.log 2>&1; rc build-$a-$m-$k $?
    s=$(ls -t python/mojolearn/identical/_mojolearn_$m.so python/mojolearn/_mojolearn_$m.so 2>/dev/null | head -1); [ -n "$s" ] && cp $s $O/$a-$m-$k.so
  done; done
done
cd $O/tree-main; base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1); $base -m venv $O/venv; P=$O/venv/bin/python
$P -m pip -q install mojolearn==0.8.33 $PLUG==0.8.33 numpy==2.5.2 scipy==1.18.0 > $O/pip.log 2>&1; $P -m pip -q install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu >> $O/pip.log 2>&1
S=$($P -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
T=$(dirname $(find $S/.. -path "*$BK/$AR/identical/_mojolearn_byte_lm.so" | head -1)); echo "$T" > $O/target.txt
cp $S/_version.py $O/_version.py
use() { a=$1; k=$2; cp $O/tree-$a/python/mojolearn/*.py $S/; cp $O/_version.py $S/_version.py; find $S -name __pycache__ -exec rm -rf {} +
  for m in $MODS; do cp $O/$a-$m-plain.so $T/_mojolearn_$m.so; done; [ $k = timers ] && cp $O/$a-byte_lm-timers.so $T/_mojolearn_byte_lm.so; true; }
st() { cd $O/tree-main; timeout 3600 $P tools/neural_stage_timing.py --lane lm-forward --lane lm-train-step --calls 8 > $O/stage-$1.log 2>&1; rc stage-$1 $?; }
for i in 1 2; do for b in $ARMS; do a=$(echo $b | tr / _); use $a plain; st $a-$i; done; done
for b in $ARMS; do a=$(echo $b | tr / _); use $a timers; st timers-$a; done
echo done > $O/done
