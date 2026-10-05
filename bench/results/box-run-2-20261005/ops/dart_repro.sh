#!/bin/bash
# dart_repro.sh <wave-dir> <vendor-backend> <gpu-arch> <column> (box-run-2): rerun the DART gate with a launch log, to a fresh dir.
W=$1; B=$2; A=$3; COL=$4; O=/root/lq/br2-dbg/dart-$(date +%s); mkdir -p $O
cd $W/on/source
env -u MOJOLEARN_IDN_ALL_OFF MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS=1 PYTHONUNBUFFERED=1 MOJOLEARN_GPU_ARCHS=$A \
  MOJOLEARN_TARGET_COLUMN=$COL MOJOLEARN_VENDOR=$B MOJOLEARN_SKIP_BUILD_GATE=1 OMP_NUM_THREADS=1 PYTHONPATH=$W/on/source/python \
  RF_LAUNCH_LOG=$O/launches.txt RF_LAUNCH_CLOCK=${CLOCK:-0} \
  /root/mojolearn/.pixi/envs/default/bin/python tools/identical_wave_dart_gate.py --data /root/board-0833/cache/algos-data/rows-full --report $O/report.json > $O/gate.log 2>&1
echo $? > $O/rc; echo $O
