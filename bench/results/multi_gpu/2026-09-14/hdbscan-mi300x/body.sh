# Two-device leg for HDBSCAN over the neighbors and hierarchy row drivers (lane/multigpu-hdbscan).
# Gates: public trace check (production), the same check against a binding built with
# -D MOJOLEARN_HIERARCHY_PARALLEL_SABOTAGE=1 (must FAIL), the native graph row gate, the unchanged HDBSCAN gate. RunPod passes no environment, so the commit is baked in. GPU work serial.
set -u
cd /root/mojolearn || exit 9
echo 55ec5f1087ad3f446f65c01f04f4789cb9b9d81a > commit.txt
OUT=/root/gemm_leg_out/hdbscan
mkdir -p "$OUT/traces"
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
job build-hdbscan $BENV sh bindings/build_hdbscan.sh &
b=$!
wait $a; wait $b
sha256sum python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_hdbscan.so > "$OUT/binaries.sha256" 2>&1
job public env PYTHONPATH=/root/mojolearn/python pixi run python tools/parallel_hdbscan_check.py --cloud --traces /root/hdb-traces --report "$OUT/public.json"
rm -rf /root/pkg-sab; mkdir -p /root/pkg-sab; cp -r python /root/pkg-sab/python
job build-hdbscan-sabotage env MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_TARGET_COLUMN=$VENDOR MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=32 MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_HIERARCHY_PARALLEL_SABOTAGE=1" sh bindings/build_hdbscan.sh
cp python/mojolearn/identical/_mojolearn_hdbscan.so /root/pkg-sab/python/mojolearn/identical/_mojolearn_hdbscan.so
sha256sum /root/pkg-sab/python/mojolearn/identical/_mojolearn_hdbscan.so >> "$OUT/binaries.sha256"
job public-sabotage env PYTHONPATH=/root/pkg-sab/python pixi run python tools/parallel_hdbscan_check.py --cloud --traces /root/hdb-traces-sab --report "$OUT/public-sabotage.json"
job graph-rows env pixi run mojo run $FLAGS -I . training/checks/graph_rows_check.mojo
job check-hdbscan pixi run mojo run $FLAGS -I . hdbscan/checks/hdbscan_check.mojo
say "public: $(tail -1 "$OUT/public.log")"
say "public-sabotage (must FAIL): $(grep -c '^PASS HDBSCAN' "$OUT/public-sabotage.log") PASS lines; $(grep -E 'AssertionError' "$OUT/public-sabotage.log" | tail -1 | cut -c1-400)"
say "graph-rows tail: $(tail -2 "$OUT/graph-rows.log" | tr '\n' '|')"
say "check-hdbscan tail: $(tail -2 "$OUT/check-hdbscan.log" | tr '\n' '|')"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
