#!/bin/bash
# NVIDIA follow-up on nvc1, after nvc1-0001: (1) an ATTN_PHASE_TIMERS byte_lm build on main+#55 for lm-forward /
# lm-train-step stage ticks; (2) toggles LAYER_SYNC=0, ATTN_SPECULATIVE=1, DEVICE_ARENA=1 on the main build;
# (3) a transformer-forward regression check: the board race on the released 0.8.25, 0.8.32 and 0.8.33 wheels.
set -uo pipefail
cd "$(dirname "$0")/.."
while [ ! -f /root/pr56-nvidia/done ]; do sleep 60; done
O=/root/nv-extra; rm -rf $O; mkdir -p $O; N=/root/neural-pass-nvidia
export PATH=/root/.pixi/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1 MOJOLEARN_COMPILE_JOBS=4 MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=sm_89 MOJOLEARN_BENCH_INSTALLED=1; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
P=$N/venv/bin/python; S=$($P -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')/mojolearn
T=$(dirname $(find $S/.. -path "*cuda/sm_89/identical/_mojolearn_byte_lm.so" | head -1))
st() { env $2 timeout 3600 $P tools/neural_stage_timing.py --lane lm-forward --lane lm-train-step --lane transformer-forward --calls 6 > $O/stage-$1.log 2>&1; rc stage-$1 $?; }
for m in transformer byte_lm training mamba linalg; do cp $N/$m.so $T/_mojolearn_$m.so; done
st main X=1
st layersync0 MOJOLEARN_BYTE_LM_LAYER_SYNC=0
st spec1 MOJOLEARN_ATTN_SPECULATIVE=1
st arena1 MOJOLEARN_DEVICE_ARENA=1
rm -f python/mojolearn/identical/_mojolearn_byte_lm.so python/mojolearn/_mojolearn_byte_lm.so
MOJOLEARN_MOJO_BUILD_FLAGS="-D MOJOLEARN_ATTN_PHASE_TIMERS=1" bash bindings/build_byte_lm.sh > $O/build-timers.log 2>&1; rc build-timers $?
so=$(ls -t python/mojolearn/identical/_mojolearn_byte_lm.so python/mojolearn/_mojolearn_byte_lm.so 2>/dev/null | head -1); [ -n "$so" ] && cp $so $T/_mojolearn_byte_lm.so
st timers X=1
cp $N/byte_lm.so $T/_mojolearn_byte_lm.so
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1)
for v in 0.8.25 0.8.32 0.8.33; do $base -m venv $O/venv-$v; Q=$O/venv-$v/bin/python
  $Q -m pip -q install mojolearn==$v mojolearn-nvidia==$v numpy==2.5.2 scipy==1.18.0 > $O/pip-$v.log 2>&1; $Q -m pip -q install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu >> $O/pip-$v.log 2>&1
  timeout 1800 $Q tools/bench_board_neural.py race --lane transformer-forward --shape full --arms ours --rounds 5 --out $O/race-tf-$v --work $O/work --ours-python $Q > $O/race-tf-$v.log 2>&1; rc race-tf-$v $?; done
echo done > $O/done
