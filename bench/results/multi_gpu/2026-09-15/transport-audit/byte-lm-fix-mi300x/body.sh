# Byte-LM transport fix gate (lane/peer-copy-root-cause): wide par-byte-lm lanes one vs two devices, and the default lanes on two devices for the 166-lane record.
set -u
cd /root/mojolearn || exit 9
echo e113ed529969b25735fab51a068c26ad20758e1a > commit.txt
OUT=/root/gemm_leg_out/lanes
mkdir -p "$OUT"
cp commit.txt "$OUT/commit.txt"
G="$OUT/gate.txt"
say() { echo "$@" | tee -a "$G"; }
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$(cat commit.txt)"
export MOJOLEARN_NUMERIC_MODE=identical
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    VENDOR=nvidia; ARCH=sm_90a; LABEL="nvidia-$ARCH"; nvidia-smi -L > "$OUT/gpus.txt"
else
    VENDOR=amd; ARCH=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+'); LABEL="amd-$ARCH"; rocm-smi --showproductname > "$OUT/gpus.txt" 2>&1
fi
say "vendor=$VENDOR arch=$ARCH"
BENV="env MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_TARGET_COLUMN=$VENDOR MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=24"
for s in build build_byte_lm build_training; do $BENV sh bindings/$s.sh > "$OUT/$s.log" 2>&1 & done
wait
for s in build build_byte_lm build_training; do say "$s: $(tail -1 "$OUT/$s.log" | cut -c1-120)"; done
LANES=par-byte-lm,par-byte-lm-model-pool,par-byte-lm-offload
IB="pixi run python tools/identity_break.py"
PY="env MOJOLEARN_COMMIT=e113ed529969b25735fab51a068c26ad20758e1a PYTHONPATH=/root/mojolearn/python"
$PY MOJOLEARN_IDENTITY_WIDE=1 MOJOLEARN_PAR_DEVICES=0 $IB --lanes $LANES --fixtures base,ties,denormal,dupes,odd --vendor "$LABEL-wide-one" --json "$OUT/wide-one.json" > "$OUT/wide-one.log" 2>&1
say "wide-one exit=$?"
$PY MOJOLEARN_IDENTITY_WIDE=1 MOJOLEARN_PAR_DEVICES=0,1 $IB --lanes $LANES --fixtures base,ties,denormal,dupes,odd --vendor "$LABEL-wide-two" --json "$OUT/wide-two.json" > "$OUT/wide-two.log" 2>&1
say "wide-two exit=$?"
$IB --diff "$OUT/wide-one.json" "$OUT/wide-two.json" > "$OUT/wide-diff.txt" 2>&1
grep -E "^summary" "$OUT/wide-diff.txt" | sed 's/^/wide /' | tee -a "$G"
$PY MOJOLEARN_PAR_DEVICES=0,1 $IB --lanes $LANES --vendor "$LABEL-default-two" --json "$OUT/default-two.json" > "$OUT/default-two.log" 2>&1
say "default-two exit=$?"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
