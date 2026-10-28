#!/bin/bash
# setup-once lane gate (compile only, plus tools/pod_smoke.sh on the CPU set).
# SET=cpu: every host family through the binding cache with declared outputs, import smoke,
#          tier_dir probe and the pod smoke CPU checks (tools/pod_smoke.sh).
# SET=sm_89|gfx942: compile every GPU root listed in gpu_roots.tsv (roots whose closure reaches
#          the changed shared modules) for that accelerator. No runtime, no timing.
set -u
OUT=${LEG_OUT:?}; SET=${SET:?}; mkdir -p "$OUT/logs"
rm -f python/mojolearn/host/*.so
if [ "$SET" = cpu ]; then
  bash tools/pod_smoke.sh; rc=$?
  echo "POD_GATE set=cpu rc=$rc" >> "$OUT/summary.txt"; exit $rc
fi
CORES=$(python3 -c 'import os;print(len(os.sched_getaffinity(0)))')
PAR=4; JOBS=$(( CORES / PAR )); [ $JOBS -lt 1 ] && JOBS=1
{ echo "set=$SET"; echo "cores_affinity=$CORES"; echo "cgroup_cpu_max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"; echo "parallel=$PAR jobs=$JOBS"; echo "mojo_version=$(pixi run -e default mojo --version 2>&1 | tail -n 1)"; } > "$OUT/resources.txt"
sh tokenizer/tools/gen_unicode_table.sh > "$OUT/logs/gen_unicode.log" 2>&1
export OUT JOBS SET
root() {
  local tag=$1 e=$2 t0=$(date +%s) tf ol="--optimization-level 0" o="$OUT/$1.so"
  case $SET in sm_89) tf="--target-accelerator sm_89 -D MOJOLEARN_COLUMN_NVIDIA";; gfx942) tf="--target-accelerator gfx942 -D MOJOLEARN_COLUMN_AMD"; ol="";; esac
  pixi run -e default mojo build $ol -j $JOBS --emit shared-lib --target-cpu x86-64-v3 $tf -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . -I bindings "$e" -o "$o" > "$OUT/logs/root_$tag.log" 2>&1
  local rc=$? sha=-; [ -f "$o" ] && sha=$(sha256sum "$o" | cut -d' ' -f1); rm -f "$o"
  printf 'ROOT\t%s\t%s\t%s\trc=%s\t%ss\tsha256=%s\n' "$SET" "$tag" "$e" "$rc" "$(( $(date +%s)-t0 ))" "$sha" >> "$OUT/roots.tsv"
}
export -f root
awk -F'\t' '{sub(/\.sh$/,"",$1); printf "root %s %s\n", $1, $2}' "${ROOTS_FILE:-experiments/setup_once_20261028/gpu_roots.tsv}" > "$OUT/root_plan.txt"
xargs -P $PAR -L 1 bash -c '"$@"' _ < "$OUT/root_plan.txt"
nr=$(wc -l < "$OUT/root_plan.txt"); okr=$(grep -c 'rc=0' "$OUT/roots.tsv" 2>/dev/null)
echo "ROOTS set=$SET total=$nr ok=$okr" >> "$OUT/summary.txt"
[ "$nr" = "$okr" ]
