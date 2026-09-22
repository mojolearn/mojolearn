# Holt-Winters estimated-initialization GPU leg (feat/holtwinters-estimated-init-sep22, 2026-09-22).
# Runs under tools/gemm_remote_leg.sh (RunPod NVIDIA) or tools/do_extra_leg.sh (DigitalOcean AMD) as
# MOJOLEARN_GEMM_LEG_EXTRA. The commit is read from the runner's own leg.txt witness, never typed.
set -u
LANES="holtwinters holtwinters-multiplicative par-holtwinters par-forecast-holtwinters"
cd /root/mojolearn || exit 9
R=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt | head -1)
[ -n "$R" ] || { echo "no commit= in leg.txt" >&2; exit 9; }
echo "$R" > /root/mojolearn/commit.txt
OUT=/root/gemm_leg_out/hwinit
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
say() { echo "$@" >> "$G"; }
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$R"
(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null; rocminfo 2>/dev/null | grep -E 'Marketing Name|^ *Name: +gfx') > "$OUT/logs/device.txt" 2>&1
say "device=$(tr '\n' ' ' < "$OUT/logs/device.txt")"
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
        case "$_cc" in 9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;; 8.9) MOJOLEARN_GPU_ARCHS=sm_89 ;; 8.6) MOJOLEARN_GPU_ARCHS=sm_86 ;; 12.0) MOJOLEARN_GPU_ARCHS=sm_120a ;; *) MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;; esac
        : "${MOJOLEARN_TARGET_COLUMN:=nvidia}"
    elif command -v rocminfo >/dev/null 2>&1; then
        MOJOLEARN_GPU_ARCHS=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+')
        : "${MOJOLEARN_TARGET_COLUMN:=amd}"
    fi
fi
export MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN
case "$MOJOLEARN_TARGET_COLUMN" in nvidia) V=nvidia ;; amd) V=amd ;; *) V=unknown ;; esac
LABEL="$V-$MOJOLEARN_GPU_ARCHS"
say "gpu_archs=$MOJOLEARN_GPU_ARCHS column=$MOJOLEARN_TARGET_COLUMN label=$LABEL"
run() { _n=$1; shift; _t0=$(date +%s); "$@" > "$OUT/logs/$_n.log" 2>&1; _e=$?; say "$_n exit=$_e secs=$(( $(date +%s) - _t0 ))"; return $_e; }
BENV="env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=8"
run check-estimate pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . holtwinters/checks/hw_estimate_check.mojo
grep -E "differing|ALL OK|FAILED" "$OUT/logs/check-estimate.log" >> "$G"
run check-identical pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . holtwinters/checks/hw_check.mojo
grep -E "ALL OK|FAILED" "$OUT/logs/check-identical.log" >> "$G"
run build-core $BENV sh bindings/build.sh
run build-tsa $BENV sh bindings/build_tsa.sh
for L in $LANES; do
    run "identity-$L" env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py --lanes "$L" --vendor "$LABEL" --repeats 1 --json "$OUT/identity_break.$LABEL.$L.json"
    grep -E "^cells=" "$OUT/logs/identity-$L.log" >> "$G"
done
run merge env PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py --merge \
    "$OUT/identity_break.$LABEL.holtwinters.json" "$OUT/identity_break.$LABEL.holtwinters-multiplicative.json" \
    "$OUT/identity_break.$LABEL.par-holtwinters.json" "$OUT/identity_break.$LABEL.par-forecast-holtwinters.json" \
    --json "$OUT/$LABEL.json"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
