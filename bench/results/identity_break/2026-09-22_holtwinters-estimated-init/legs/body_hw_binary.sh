# Combined GPU leg, 2026-09-22: the Holt-Winters estimated-initialization lanes
# (body_hw.sh, same commands) AND the gbdt-binary-columns lane
# (bench/results/identity_break/2026-09-22_gbdt-binary-columns/README.md), one
# rental per vendor. Runs under tools/hotaisle_leg.sh, tools/do_extra_leg.sh or
# tools/gemm_remote_leg.sh as MOJOLEARN_GEMM_LEG_EXTRA. The commit is read from
# the runner's own leg.txt witness, never typed.
#
# Order: the recordings first (they are what is owed), the two Mojo gates of
# body_hw.sh last, so a lease that runs short still brings the columns home.
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
T0=$(date +%s)
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
case "$MOJOLEARN_TARGET_COLUMN" in nvidia) V=nvidia; BK=cuda ;; amd) V=amd; BK=hip ;; *) V=unknown; BK=none ;; esac
LABEL="$V-$MOJOLEARN_GPU_ARCHS"
# the card's model for the gbdt-binary-columns vendor label (nvidia-<gpu>-binary-columns)
_dev=$(tr 'A-Z' 'a-z' < "$OUT/logs/device.txt")
case "$_dev" in
    *h100*) GPUN=h100 ;; *h200*) GPUN=h200 ;; *4090*) GPUN=rtx4090 ;; *5090*) GPUN=rtx5090 ;;
    *a100*) GPUN=a100 ;; *l40s*) GPUN=l40s ;; *mi300x*) GPUN=mi300x ;; *mi325x*) GPUN=mi325x ;;
    *) GPUN="$MOJOLEARN_GPU_ARCHS" ;;
esac
BLABEL="$V-$GPUN-binary-columns"
say "gpu_archs=$MOJOLEARN_GPU_ARCHS column=$MOJOLEARN_TARGET_COLUMN label=$LABEL gbdt_label=$BLABEL backend=$BK"
run() { _n=$1; shift; _t0=$(date +%s); "$@" > "$OUT/logs/$_n.log" 2>&1; _e=$?; say "$_n exit=$_e secs=$(( $(date +%s) - _t0 ))"; return $_e; }

BINCACHE="sh"
if command -v python3 > /dev/null 2>&1 && [ -f tools/bincache.py ]; then
    BINCACHE="python3 tools/bincache.py build"
fi
if [ -f /root/.mojolearn_bincache/urls.tsv ]; then
    say "bincache map staged entries=$(grep -c '^get' /root/.mojolearn_bincache/urls.tsv)"
else
    say "bincache map absent (every build compiles from source)"
fi
# shellcheck disable=SC2086  # $BINCACHE is one word or three, split on purpose
build() { run "$1" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=8 $BINCACHE "bindings/$1.sh"; }

# `import mojolearn` needs python/mojolearn/.libs/libMojolearnMath.so, which
# git archive does not ship (tools/gpu_class_gaps_amd_leg.sh): the tree's own recipe.
run portable_math env PYTHONPATH=/root/mojolearn/packaging/portable_math \
    pixi run python -c "import pathlib, stage; stage.build(pathlib.Path('/root/mojolearn/python/mojolearn/.libs/libMojolearnMath.so'))"
build build
build build_tsa
build build_gbdt
sha256sum python/mojolearn/*.so python/mojolearn/identical/*.so >> "$G" 2>/dev/null
run import_probe env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python -c "import mojolearn; print('IMPORT_OK', mojolearn.vendor(), mojolearn.__version__)"
say "import_probe=$(tail -1 "$OUT/logs/import_probe.log" 2>/dev/null)"

# ---- B. gbdt-binary-columns (the README's command, verbatim but for the paths)
run identity-gbdt-binary-columns env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --require-backend "$BK" --lanes gbdt-binary-columns --repeats 1 \
    --batch-grad --batch-scale --ragged --step-full \
    --vendor "$BLABEL" --json "$OUT/$V-$GPUN.json"
grep -E "^cells=|^train:|^infer:|^model:|^batch" "$OUT/logs/identity-gbdt-binary-columns.log" >> "$G"
say "elapsed_after_gbdt=$(( $(date +%s) - T0 ))"

# ---- A. the four Holt-Winters lanes (body_hw.sh)
for L in $LANES; do
    run "identity-$L" env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py --lanes "$L" --vendor "$LABEL" --repeats 1 --json "$OUT/identity_break.$LABEL.$L.json"
    grep -E "^cells=" "$OUT/logs/identity-$L.log" >> "$G"
done
run merge env PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py --merge \
    "$OUT/identity_break.$LABEL.holtwinters.json" "$OUT/identity_break.$LABEL.holtwinters-multiplicative.json" \
    "$OUT/identity_break.$LABEL.par-holtwinters.json" "$OUT/identity_break.$LABEL.par-forecast-holtwinters.json" \
    --json "$OUT/$LABEL.json"
say "elapsed_after_hw=$(( $(date +%s) - T0 ))"

# ---- the Mojo gates of body_hw.sh, last
run check-estimate pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . holtwinters/checks/hw_estimate_check.mojo
grep -E "differing|ALL OK|FAILED" "$OUT/logs/check-estimate.log" >> "$G"
run check-identical pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . holtwinters/checks/hw_check.mojo
grep -E "ALL OK|FAILED" "$OUT/logs/check-identical.log" >> "$G"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) elapsed=$(( $(date +%s) - T0 ))"
