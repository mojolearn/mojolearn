# Two-device leg for KernelRidge/Nystroem kernel rows + Cholesky rows/columns and RBFSampler query rows
# (lane/multigpu-kernel-methods, built on lane/multigpu-cholesky). Vendor detected from the box.
# Gates: public check (production build), the same public check against a build with
# -D MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE=1 (must FAIL), the Cholesky native gate, and the unchanged
# single-device kernel-methods gate. RunPod passes no environment, so the commit is baked in. GPU work serial.
set -u
cd /root/mojolearn || exit 9
echo f229b8b3cfa0d81e4b9e7ba64ff5936439f8e347 > commit.txt
OUT=/root/gemm_leg_out/km
mkdir -p "$OUT/chol-traces"
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
job build-km $BENV sh bindings/build_kernel_methods.sh &
b=$!
wait $a; wait $b
sha256sum python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_kernel_methods.so > "$OUT/binaries.sha256" 2>&1
job public env PYTHONPATH=/root/mojolearn/python pixi run python tools/parallel_kernel_methods_check.py --cloud --report "$OUT/public.json"
rm -rf /root/pkg-sab; mkdir -p /root/pkg-sab; cp -r python /root/pkg-sab/python
job build-km-sabotage env MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_TARGET_COLUMN=$VENDOR MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=32 MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE=1" sh bindings/build_kernel_methods.sh
cp python/mojolearn/identical/_mojolearn_kernel_methods.so /root/pkg-sab/python/mojolearn/identical/_mojolearn_kernel_methods.so
sha256sum /root/pkg-sab/python/mojolearn/identical/_mojolearn_kernel_methods.so >> "$OUT/binaries.sha256"
job public-sabotage env PYTHONPATH=/root/pkg-sab/python pixi run python tools/parallel_kernel_methods_check.py --cloud --report "$OUT/public-sabotage.json"
job native-cholesky env MOJOLEARN_CHOLESKY_CHECK_DIR="$OUT/chol-traces" pixi run mojo run $FLAGS -I . training/checks/cholesky_parallel_check.mojo
job native-cholesky-sabotage-factor env MOJOLEARN_CHOLESKY_CHECK_FACTOR_ONLY=1 MOJOLEARN_CHOLESKY_CHECK_DIR="$OUT/chol-traces" pixi run mojo run $FLAGS -D MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE=1 -I . training/checks/cholesky_parallel_check.mojo
job check-kernel-methods pixi run mojo run $FLAGS -I . kernel_methods/checks/km_check.mojo
say "public: $(tail -1 "$OUT/public.log")"
say "public-sabotage (must FAIL): $(grep -c '^PASS' "$OUT/public-sabotage.log") PASS lines; $(grep -E 'AssertionError|Error' "$OUT/public-sabotage.log" | tail -2 | tr '\n' '|')"
say "native-cholesky: $(grep -c '^PASS cholesky' "$OUT/native-cholesky.log") PASS lines; last: $(grep '^PASS cholesky parallel gate' "$OUT/native-cholesky.log")"
say "native-cholesky-sabotage-factor (must FAIL): $(grep -c '^PASS cholesky' "$OUT/native-cholesky-sabotage-factor.log") PASS lines; $(grep 'Unhandled' "$OUT/native-cholesky-sabotage-factor.log" | tail -1)"
say "check-kernel-methods tail: $(tail -3 "$OUT/check-kernel-methods.log" | tr '\n' '|')"
rm -rf "$OUT/chol-traces"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
