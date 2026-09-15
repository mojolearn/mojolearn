# Two-MI300X leg for the pooled RF/ET grove predictor (lane/forest-grove-pool).
# Same gates as bench/results/multi_gpu/2026-09-14/forest-grove-pool-h100: native pool check
# (production and fault builds, separate and packed layouts), resident model lifecycle check, and
# the public ParallelForestPredictor check against single-device parallel_groves.
# RunPod passes no environment, so the commit is baked in.
set -u
cd /root/mojolearn || exit 9
echo 81120c4edf5d13cf7451d8cb135648782b09ed49 > commit.txt
OUT=/root/gemm_leg_out/forest
mkdir -p "$OUT"
G="$OUT/gate.txt"
say() { echo "$@" | tee -a "$G"; }
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$(cat commit.txt)"
export RUNPOD_POD_ID="${RUNPOD_POD_ID:-runpod-amd-leg}"
export MOJOLEARN_NUMERIC_MODE=identical
rocm-smi --showproductname > "$OUT/gpus.txt" 2>&1
rocminfo 2>/dev/null | grep -E 'Marketing Name|^ *Name: +gfx' >> "$OUT/gpus.txt"
ARCH=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+')
NGPU=$(rocminfo 2>/dev/null | grep -cE '^ *Name: +gfx')
say "arch=$ARCH gpus=$NGPU nproc=$(nproc) mem=$(free -g | awk '/Mem:/{print $2}')G"
[ "$NGPU" -ge 2 ] || { say "FATAL fewer than two GPUs"; exit 0; }
FLAGS="--target-accelerator $ARCH -D MOJOLEARN_COLUMN_AMD -D MOJOLEARN_NUMERIC_IDENTICAL=1"
PACK="-D MOJOLEARN_FOREST_PACKED_NODES=1"
FAULT="-D MOJOLEARN_FOREST_POOL_FAULT=1"
st() { echo "$1 exit=$2 seconds=$3" | tee -a "$G"; }
job() { # name, command...
    _n=$1; shift; _t=$(date +%s)
    "$@" > "$OUT/$_n.log" 2>&1; _e=$?
    st "$_n" "$_e" "$(( $(date +%s) - _t ))"
}
build_layout() { # layout extra
    _x=$2
    job "build-rf-$1" env MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_TARGET_COLUMN=amd MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=16 MOJOLEARN_EXTRA_DEFINES="$_x" sh bindings/build_rf.sh &
    _a=$!
    job "build-et-$1" env MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_TARGET_COLUMN=amd MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=16 MOJOLEARN_EXTRA_DEFINES="$_x" sh bindings/build_trees.sh &
    _b=$!
    wait $_a; wait $_b
    rm -rf "/root/pkg-$1"; mkdir -p "/root/pkg-$1"; cp -r python "/root/pkg-$1/python"
    sha256sum "/root/pkg-$1/python/mojolearn/identical/_mojolearn_rf.so" "/root/pkg-$1/python/mojolearn/identical/_mojolearn_trees.so" >> "$OUT/binaries.sha256"
}
native() { # name extra
    job "$1" pixi run mojo run $FLAGS $2 -I . training/checks/forest_pool_check.mojo
}
# Warm pixi once so parallel runs do not race the environment install.
job pixi-warm pixi run python -c "import numpy; print(numpy.__version__)"
# Phase A: separate-layout bindings, plus both production native checks.
native native-separate "" &
na=$!
native native-packed "$PACK" &
nb=$!
build_layout separate ""
wait $na; wait $nb
# Phase B: packed bindings; meanwhile the public separate check (own package copy) and fault builds.
public() {
    job "public-$1" env PYTHONPATH=/root/pkg-$1/python pixi run python tools/parallel_forest_pool_check.py --cloud --report "$OUT/public-$1.json"
}
public separate &
pa=$!
native fault-separate "$FAULT" &
fa=$!
native fault-packed "$FAULT $PACK" &
fb=$!
build_layout packed "$PACK"
wait $pa; wait $fa; wait $fb
public packed &
pb=$!
job resident-separate pixi run mojo run $FLAGS -I . checks/forest_inference_model.mojo &
ra=$!
job resident-packed pixi run mojo run $FLAGS $PACK -I . checks/forest_inference_model.mojo &
rb=$!
wait $pb; wait $ra; wait $rb
for f in native-separate native-packed fault-separate fault-packed; do
    say "$f PASS_lines=$(grep -c '^PASS forest pool' "$OUT/$f.log") witness=$(grep -c 'serial-order witness' "$OUT/$f.log")"
done
for f in public-separate public-packed resident-separate resident-packed; do say "$f: $(grep -E '^PASS|PASS$|_PASS|Error|error:' "$OUT/$f.log" | tail -3 | tr '\n' '|')"; done
python3 - "$OUT" <<'PY' | tee -a "$G"
import json, sys
from pathlib import Path
r = Path(sys.argv[1])
try:
    a = json.loads((r/'public-separate.json').read_text()); b = json.loads((r/'public-packed.json').read_text())
    print('public separate==packed', a == b)
except Exception as e:
    print('public compare failed', e)
PY
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
