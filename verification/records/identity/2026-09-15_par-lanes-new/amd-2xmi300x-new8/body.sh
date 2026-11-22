# identity_break par lanes on one device and on two devices of the same box (lane/peer-copy-root-cause).
set -u
cd /root/mojolearn || exit 9
echo e520c8f49d6330ba832dfefcb552805cc733644d > commit.txt
OUT=/root/gemm_leg_out/lanes
mkdir -p "$OUT"
cp commit.txt "$OUT/commit.txt"
G="$OUT/gate.txt"
say() { echo "$@" | tee -a "$G"; }
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$(cat commit.txt)"
export MOJOLEARN_NUMERIC_MODE=identical
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    VENDOR=nvidia; _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
    case "$_cc" in 9.0) ARCH=sm_90a ;; 8.9) ARCH=sm_89 ;; 12.0) ARCH=sm_120a ;; *) ARCH="sm_$(echo "$_cc" | tr -d .)" ;; esac
    LABEL="nvidia-$ARCH"; nvidia-smi -L > "$OUT/gpus.txt"
else
    VENDOR=amd; ARCH=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+'); LABEL="amd-$ARCH"; rocm-smi --showproductname > "$OUT/gpus.txt" 2>&1
fi
say "vendor=$VENDOR arch=$ARCH"
BENV="env MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_TARGET_COLUMN=$VENDOR MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=24"
for s in build build_gp build_kernel_methods build_mixture build_resample build_hdbscan build_ivf build_embedding; do $BENV sh bindings/$s.sh > "$OUT/$s.log" 2>&1 & done
wait
for s in build build_gp build_kernel_methods build_mixture build_resample build_hdbscan build_ivf build_embedding; do say "$s: $(tail -1 "$OUT/$s.log" | cut -c1-160)"; done
say "builds: $(ls python/mojolearn/identical/*.so | wc -l) identical bindings"
LANES=cholesky,kernel-ridge,nystroem,rbf-sampler,par-cholesky,par-kernel-ridge,par-nystroem,par-rbf-sampler
env MOJOLEARN_PAR_DEVICES=0 MOJOLEARN_COMMIT=e520c8f49d6330ba832dfefcb552805cc733644d PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py --lanes "$LANES" --vendor "$LABEL-one" --json "$OUT/one.json" > "$OUT/one.log" 2>&1
say "one exit=$? finished=$(date -u +%H:%M:%SZ)"
env MOJOLEARN_PAR_DEVICES=0,1 MOJOLEARN_COMMIT=e520c8f49d6330ba832dfefcb552805cc733644d PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py --lanes "$LANES" --vendor "$LABEL-two" --json "$OUT/two.json" > "$OUT/two.log" 2>&1
say "two exit=$?"
pixi run python tools/identity_break.py --diff "$OUT/one.json" "$OUT/two.json" > "$OUT/diff.txt" 2>&1
say "diff exit=$?"
grep -E "summary|REFUSED|DIVERGENT|MOVED|ERROR" "$OUT/diff.txt" | head -30 | tee -a "$G"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
