# par-forest-pool and par-gmm lanes on one device and on two devices of the same box (lane/multigpu-par-lanes, third run).
set -u
cd /root/mojolearn || exit 9
echo f067bbbc0d8f863e30acb12f4c90fe3de1d88709 > commit.txt
OUT=/root/gemm_leg_out/lanes
mkdir -p "$OUT"
G="$OUT/gate.txt"
say() { echo "$@" | tee -a "$G"; }
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$(cat commit.txt)"
export MOJOLEARN_NUMERIC_MODE=identical
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    VENDOR=nvidia; _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
    case "$_cc" in 9.0) ARCH=sm_90a ;; 8.9) ARCH=sm_89 ;; 12.0) ARCH=sm_120a ;; *) ARCH="sm_$(echo "$_cc" | tr -d .)" ;; esac
    LABEL="nvidia-h100-$ARCH"
else
    VENDOR=amd; ARCH=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+'); LABEL="amd-$ARCH"
fi
say "vendor=$VENDOR arch=$ARCH"
BENV="env MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_TARGET_COLUMN=$VENDOR MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=24"
for s in build build_rf build_trees build_mixture build_resample build_hdbscan; do $BENV sh bindings/$s.sh > "$OUT/$s.log" 2>&1 & done
wait
say "builds: $(ls python/mojolearn/identical/*.so | wc -l) identical bindings"
LANES=rf-clf,gmm,bootstrap,hdbscan,par-forest,par-forest-pool,par-gmm,par-resample,par-hdbscan
env MOJOLEARN_PAR_DEVICES=0 MOJOLEARN_COMMIT=f067bbbc0d8f863e30acb12f4c90fe3de1d88709 PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py --lanes "$LANES" --vendor "$LABEL-one" --json "$OUT/one.json" > "$OUT/one.log" 2>&1
say "one exit=$?"
env MOJOLEARN_PAR_DEVICES=0,1 MOJOLEARN_COMMIT=f067bbbc0d8f863e30acb12f4c90fe3de1d88709 PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py --lanes "$LANES" --vendor "$LABEL-two" --json "$OUT/two.json" > "$OUT/two.log" 2>&1
say "two exit=$?"
pixi run python tools/identity_break.py --diff "$OUT/one.json" "$OUT/two.json" > "$OUT/diff.txt" 2>&1
say "diff exit=$?"
grep -E "summary|REFUSED|DIVERGENT|MOVED" "$OUT/diff.txt" | head -20 | tee -a "$G"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
