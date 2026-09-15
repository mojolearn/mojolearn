# tools/transport_audit_check.py on a two-GPU box (lane/peer-copy-root-cause).
set -u
cd /root/mojolearn || exit 9
echo 3250fcb5b37e0161585793ca0bcba6d6ca2d05ef > commit.txt
OUT=/root/gemm_leg_out/audit
mkdir -p "$OUT"
cp commit.txt "$OUT/commit.txt"
G="$OUT/gate.txt"
say() { echo "$@" | tee -a "$G"; }
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$(cat commit.txt)"
export MOJOLEARN_NUMERIC_MODE=identical RUNPOD_POD_ID="${RUNPOD_POD_ID:-runpod-leg}"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    VENDOR=nvidia; ARCH=sm_90a; nvidia-smi -L > "$OUT/gpus.txt"
else
    VENDOR=amd; ARCH=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+'); rocm-smi --showproductname > "$OUT/gpus.txt" 2>&1
fi
say "vendor=$VENDOR arch=$ARCH"
BENV="env MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_TARGET_COLUMN=$VENDOR MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=16"
for s in build build_estimators build_solver build_svm build_gp build_hdbscan build_rf build_trees; do $BENV sh bindings/$s.sh > "$OUT/$s.log" 2>&1 & done
wait
say "builds: $(ls python/mojolearn/identical/*.so | wc -l) identical bindings"
env PYTHONPATH=/root/mojolearn/python timeout 1800 pixi run python tools/transport_audit_check.py > "$OUT/audit.log" 2>&1
say "audit exit=$?"
env PYTHONPATH=/root/mojolearn/python MOJOLEARN_TRANSPORT_AUDIT_CASES=cd,gp MOJOLEARN_TRANSPORT_AUDIT_SABOTAGE=1 timeout 300 pixi run python tools/transport_audit_check.py > "$OUT/audit-sabotage.log" 2>&1
say "audit-sabotage (must exit 1) exit=$?"
grep -h "^TRANSPORT" "$OUT/audit.log" "$OUT/audit-sabotage.log" | tee -a "$G"
tail -5 "$OUT/audit.log" >> "$G"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
