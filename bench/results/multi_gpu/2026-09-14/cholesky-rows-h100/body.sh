# Two-device leg for the operation-level Cholesky rows and right-hand sides (lane/multigpu-cholesky). Vendor detected from the box.
# Gates: native E-step/full-fit/trace equality (production), the same gate under the sabotage build (must FAIL),
# the unchanged single-device Cholesky gate, and the public fit_cholesky/solve_cholesky check against one device.
# RunPod passes no environment, so the commit is baked in. Every GPU process runs serially.
set -u
cd /root/mojolearn || exit 9
echo 00037ee3781604afe9ba6f65df5bc13e60924791 > commit.txt
OUT=/root/gemm_leg_out/cholesky
mkdir -p "$OUT/traces" "$OUT/traces-sab"
G="$OUT/gate.txt"
say() { echo "$@" | tee -a "$G"; }
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$(cat commit.txt)"
export RUNPOD_POD_ID="${RUNPOD_POD_ID:-runpod-leg}"
export MOJOLEARN_NUMERIC_MODE=identical
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    VENDOR=nvidia; COL=NVIDIA
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv > "$OUT/gpus.txt" 2>&1
    _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
    case "$_cc" in 9.0) ARCH=sm_90a ;; 8.9) ARCH=sm_89 ;; 12.0) ARCH=sm_120a ;; *) ARCH="sm_$(echo "$_cc" | tr -d .)" ;; esac
    NGPU=$(nvidia-smi -L | wc -l)
else
    VENDOR=amd; COL=AMD
    rocm-smi --showproductname > "$OUT/gpus.txt" 2>&1
    ARCH=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+')
    NGPU=$(rocminfo 2>/dev/null | grep -cE '^ *Name: +gfx')
fi
say "vendor=$VENDOR arch=$ARCH gpus=$NGPU nproc=$(nproc)"
[ "$NGPU" -ge 2 ] || { say "FATAL fewer than two GPUs"; exit 0; }
FLAGS="--target-accelerator $ARCH -D MOJOLEARN_COLUMN_$COL -D MOJOLEARN_NUMERIC_IDENTICAL=1"
st() { echo "$1 exit=$2 seconds=$3" | tee -a "$G"; }
job() {
    _n=$1; shift; _t=$(date +%s)
    "$@" > "$OUT/$_n.log" 2>&1; _e=$?
    st "$_n" "$_e" "$(( $(date +%s) - _t ))"
    return $_e
}
BENV="env MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_TARGET_COLUMN=$VENDOR MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=32"
job pixi-warm pixi run python -c "import numpy; print(numpy.__version__)"
job build-base $BENV sh bindings/build.sh &
a=$!
job build-gp $BENV sh bindings/build_gp.sh &
b=$!
wait $a; wait $b
sha256sum python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_gp.so > "$OUT/binaries.sha256" 2>&1
job native env MOJOLEARN_CHOLESKY_CHECK_DIR="$OUT/traces" pixi run mojo run $FLAGS -I . training/checks/cholesky_parallel_check.mojo
job native-sabotage env MOJOLEARN_CHOLESKY_CHECK_DIR="$OUT/traces-sab" pixi run mojo run $FLAGS -D MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE=1 -I . training/checks/cholesky_parallel_check.mojo
job public env PYTHONPATH=/root/mojolearn/python pixi run python tools/parallel_cholesky_check.py --cloud --report "$OUT/public.json"
job check-cholesky pixi run mojo run $FLAGS -I . cholesky/checks/cholesky_check.mojo
say "native: $(grep -c '^PASS cholesky' "$OUT/native.log") PASS lines; last: $(grep -E '^PASS cholesky parallel gate' "$OUT/native.log")"
say "native-sabotage (must FAIL): $(grep -c '^PASS cholesky' "$OUT/native-sabotage.log") PASS lines; $(grep -E 'Error|error:' "$OUT/native-sabotage.log" | grep -v warning | tail -2 | tr '\n' '|')"
say "public: $(tail -1 "$OUT/public.log")"
say "check-cholesky: $(grep -ciE '^PASS|  PASS|ok' "$OUT/check-cholesky.log") pass-like lines; tail: $(tail -2 "$OUT/check-cholesky.log" | tr '\n' '|')"
# Trace summaries only: per-fit record counts and one sha256 per trace file (the raw traces stay on the box).
( cd "$OUT/traces" && for f in *.trace; do echo "$(sha256sum "$f" | cut -c1-64) $(grep -vc '^#' "$f") $f"; done ) > "$OUT/trace_digests.txt" 2>&1
rm -rf "$OUT/traces-sab"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
