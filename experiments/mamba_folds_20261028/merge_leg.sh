#!/bin/bash
# Self-merge pod leg for lane mamba-folds (COMMON.md "Self-merge to main"): compiles, scripted gates, pod smoke.
set -u
mkdir -p "$LEG_OUT"
CORES=$(python3 -c 'import os;print(len(os.sched_getaffinity(0)))'); echo "cores=$CORES" > "$LEG_OUT/host.txt"
export DEFSETS="off bwd" MAIN=$PWD JOBS=$CORES
bash experiments/mamba_folds_20261028/compile.sh "$LEG_OUT/cpu" host; c1=$?
COLUMN_DEFS="-D MOJOLEARN_COLUMN_NVIDIA" EXTRA="--target-accelerator sm_89" bash experiments/mamba_folds_20261028/compile.sh "$LEG_OUT/sm89" device training; c2=$?
for f in "$LEG_OUT"/sm89/*.so; do echo "$(basename $f) sm_89_strings=$(strings "$f" | grep -c sm_89)"; done >> "$LEG_OUT/host.txt" 2>/dev/null
rm -f "$LEG_OUT"/cpu/*.so "$LEG_OUT"/sm89/*.so
pixi run -e default python tools/tests/test_no_python_runtime.py > "$LEG_OUT/gate_no_python.log" 2>&1; g1=$?
pixi run -e default python tools/tests/test_tombstones_preserved.py > "$LEG_OUT/gate_tombstones.log" 2>&1; g2=$?
python3 tools/mamba_host_gen.py --check > "$LEG_OUT/gate_hostgen.log" 2>&1; g3=$?
LEG_OUT="$LEG_OUT/smoke" bash experiments/mamba_folds_20261028/pod_smoke.sh > "$LEG_OUT/smoke.out" 2>&1; s1=$?
echo "LEG compile_cpu=$c1 compile_sm89=$c2 no_python=$g1 tombstones=$g2 hostgen=$g3 smoke=$s1" | tee -a "$LEG_OUT/host.txt"
[ $c1 = 0 ] && [ $c2 = 0 ] && [ $g1 = 0 ] && [ $g2 = 0 ] && [ $g3 = 0 ] && [ $s1 = 0 ]
