# Cholesky trailing-row transport fix gate (lane/peer-copy-root-cause).
set -u
cd /root/mojolearn || exit 9
echo ce4634f4d52bf6d694df940aff868d2cee20700d > commit.txt
OUT=/root/gemm_leg_out/audit
mkdir -p "$OUT/chol-traces"
cp commit.txt "$OUT/commit.txt"
G="$OUT/gate.txt"
say() { echo "$@" | tee -a "$G"; }
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$(cat commit.txt)"
export MOJOLEARN_NUMERIC_MODE=identical RUNPOD_POD_ID="${RUNPOD_POD_ID:-runpod-leg}"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    VENDOR=nvidia; COL=NVIDIA; ARCH=sm_90a; nvidia-smi -L > "$OUT/gpus.txt"
else
    VENDOR=amd; COL=AMD; ARCH=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+'); rocm-smi --showproductname > "$OUT/gpus.txt" 2>&1
fi
say "vendor=$VENDOR arch=$ARCH"
BENV="env MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_TARGET_COLUMN=$VENDOR MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=16"
for s in build build_estimators build_linalg build_gp build_kernel_methods; do $BENV sh bindings/$s.sh > "$OUT/$s.log" 2>&1 & done
wait
say "builds: $(ls python/mojolearn/identical/*.so | wc -l) identical bindings"
env PYTHONPATH=/root/mojolearn/python MOJOLEARN_TRANSPORT_AUDIT_CASES=cholesky timeout 900 pixi run python tools/transport_audit_check.py > "$OUT/audit.log" 2>&1
say "audit cholesky exit=$?"
env PYTHONPATH=/root/mojolearn/python MOJOLEARN_TRANSPORT_AUDIT_CASES=cholesky timeout 900 pixi run python tools/transport_audit_check.py > "$OUT/audit-repeat.log" 2>&1
say "audit cholesky repeat exit=$?"
grep -h "^TRANSPORT" "$OUT/audit.log" "$OUT/audit-repeat.log" | tee -a "$G"
env MOJOLEARN_CHOLESKY_CHECK_DIR="$OUT/chol-traces" timeout 1200 pixi run mojo run --target-accelerator $ARCH -D MOJOLEARN_COLUMN_$COL -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/cholesky_parallel_check.mojo > "$OUT/native-cholesky.log" 2>&1
say "native cholesky exit=$? $(tail -1 "$OUT/native-cholesky.log")"
( cd "$OUT/chol-traces" && for f in *.trace; do echo "$(sha256sum "$f" | cut -c1-64) $f"; done ) > "$OUT/chol_trace_digests.txt" 2>&1
env MOJOLEARN_COMMIT=ce4634f4d52bf6d694df940aff868d2cee20700d MOJOLEARN_PAR_DEVICES=0,1 PYTHONPATH=/root/mojolearn/python timeout 1200 pixi run python tools/identity_break.py --lanes par-cholesky,par-kernel-ridge --vendor "$VENDOR-two" --json "$OUT/two.json" > "$OUT/two.log" 2>&1
say "par lanes two exit=$?"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
