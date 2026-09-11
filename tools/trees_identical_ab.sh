#!/bin/sh
# RUNS ON THE POD. The trees lane's IDENTICAL-tier A/B helper: named binary
# sets under /root/bins/<set>/, swapped into python/mojolearn/identical/
# before each run, so one checkout serves every arm.
#
#   sh tools/trees_identical_ab.sh build <set> <binding> [extra -D defines]
#       binding: base | gbdt | rf | trees; the other three .so are copied
#       from /root/bins/baseline so <set> is always a complete tier.
#   sh tools/trees_identical_ab.sh use <set>
#   sh tools/trees_identical_ab.sh ib <set> [lanes]      identity_break fingerprints
#   sh tools/trees_identical_ab.sh diff <setA> <setB>    identity_break --diff
#   sh tools/trees_identical_ab.sh speed <set> <lane> <dataset> <rows> <rounds> [full|ours|stage]
#       full (default) interleaves ours with the opponents; ours times our arm
#       alone (the A/B between two of our builds); stage runs ONE untimed
#       ours-only replicate with MOJOLEARN_STAGE_TIMES=1
#   sh tools/trees_identical_ab.sh rfgate <set> [extra -D defines]
#       ensemble/checks/rf_perf_candidates_check.mojo under the identical define
set -u
ROOT=/root/mojolearn
OUT=/root/trees_out
BINS=/root/bins
LOGS="$OUT/logs"
mkdir -p "$LOGS" "$BINS" "$OUT/ib" "$OUT/speed"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
# Vendor from the box (2026-09-11, the AMD leg): NVIDIA -> cuda, else AMD -> hip.
if command -v nvidia-smi > /dev/null 2>&1; then
    export MOJOLEARN_SPEED_EXPECTED_VENDOR=cuda
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')"
    IB_VENDOR="nvidia-$GPU_NAME"
else
    export MOJOLEARN_SPEED_EXPECTED_VENDOR=hip
    GPU_NAME="$(rocm-smi --showproductname 2>/dev/null | sed -n 's/.*Card Series:[[:space:]]*//p' | head -1 | tr ' ' '_')"
    IB_VENDOR="amd-${GPU_NAME:-unknown}"
fi
# The interpreter (a venv carrying AMD's ROCm xgboost, say); python3 by default.
PY="${MOJOLEARN_SPEED_PY:-python3}"
TIER=python/mojolearn/identical

cmd_build() {
    _set="$1"; _b="$2"; shift 2
    mkdir -p "$BINS/$_set"
    for f in "$BINS/baseline"/*.so; do [ -f "$BINS/$_set/$(basename "$f")" ] || cp "$f" "$BINS/$_set/"; done
    case "$_b" in
        base) _script=bindings/build.sh; _so=_mojolearn.so ;;
        gbdt) _script=bindings/build_gbdt.sh; _so=_mojolearn_gbdt.so ;;
        rf) _script=bindings/build_rf.sh; _so=_mojolearn_rf.so ;;
        trees) _script=bindings/build_trees.sh; _so=_mojolearn_trees.so ;;
        *) echo "unknown binding $_b"; exit 2 ;;
    esac
    echo "build $_set/$_b defines='$*' $(date -u +%H:%M:%S)"
    MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=4 MOJOLEARN_EXTRA_DEFINES="$*" \
        timeout -k 30 1500 bash "$_script" > "$LOGS/build.$_set.$_b.log" 2>&1
    _rc=$?
    echo "build_exit $_set.$_b=$_rc $(date -u +%H:%M:%S)" | tee -a "$OUT/ab.txt"
    if [ "$_rc" = 0 ] && [ -f "$TIER/$_so" ]; then
        cp "$TIER/$_so" "$BINS/$_set/$_so"
        sha256sum "$BINS/$_set/$_so" | tee -a "$OUT/ab.txt"
    else
        grep -m5 -i 'error' "$LOGS/build.$_set.$_b.log"
    fi
    return $_rc
}

cmd_use() {
    _set="$1"
    [ -d "$BINS/$_set" ] || { echo "no set $_set"; exit 2; }
    mkdir -p "$TIER"
    rm -f "$TIER"/*.so
    cp "$BINS/$_set"/*.so "$TIER/"
    echo "$_set" > "$OUT/current_set.txt"
}

cmd_ib() {
    _set="$1"; _lanes="${2:-rf-clf,rf-reg,et-clf,et-reg,gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse,kmeans}"
    cmd_use "$_set"
    echo "identity_break $_set lanes=$_lanes $(date -u +%H:%M:%S)"
    PYTHONPATH="$ROOT/python" timeout -k 30 1800 $PY -u tools/identity_break.py \
        --lanes "$_lanes" --vendor "$IB_VENDOR" \
        --json "$OUT/ib/$_set.json" > "$OUT/ib/$_set.txt" 2>&1
    echo "ib_exit $_set=$? $(date -u +%H:%M:%S)" | tee -a "$OUT/ab.txt"
    tail -3 "$OUT/ib/$_set.txt"
}

cmd_diff() {
    PYTHONPATH="$ROOT/python" python3 tools/identity_break.py --diff "$OUT/ib/$1.json" "$OUT/ib/$2.json" \
        > "$OUT/ib/diff.$1.$2.txt" 2>&1
    echo "ib_diff $1 vs $2 exit=$? $(date -u +%H:%M:%S)" | tee -a "$OUT/ab.txt"
    grep -c IDENTICAL "$OUT/ib/diff.$1.$2.txt"; grep -E 'DIVERGENT|MOVED|REFUSED' "$OUT/ib/diff.$1.$2.txt" | head
}

cmd_speed() {
    _set="$1"; _lane="$2"; _ds="$3"; _rows="$4"; _rounds="$5"; _mode="${6:-full}"
    cmd_use "$_set"
    export MOJOLEARN_SPEED_SIZE=shipped MOJOLEARN_SPEED_BUDGET_S=1800 MOJOLEARN_SPEED_DEADLINE_S=3600
    export MOJOLEARN_SPEED_DEVICE="$GPU_NAME"
    # Optional: MOJOLEARN_SPEED_TAG suffixes the log name (a second pass);
    # MOJOLEARN_SPEED_DEVICES / _ARMS / _OPPONENTS_FIRST=1 reach the harness
    # as --devices / --arms / --opponents-first (full mode only);
    # MOJOLEARN_SPEED_SMI_SAMPLE=1 samples rocm-smi GPU use every 5 s beside
    # the log (the proof that a GPU arm ran on the GPU).
    _log="$OUT/speed/$_set.$_lane.$_ds.r$_rows.$_mode${MOJOLEARN_SPEED_TAG:+.$MOJOLEARN_SPEED_TAG}.log"
    _extra="--devices ${MOJOLEARN_SPEED_DEVICES:-auto}"
    [ -n "${MOJOLEARN_SPEED_ARMS:-}" ] && _extra="$_extra --arms $MOJOLEARN_SPEED_ARMS"
    [ "${MOJOLEARN_SPEED_OPPONENTS_FIRST:-0}" = 1 ] && _extra="$_extra --opponents-first"
    # MOJOLEARN_SPEED_OURS_AB=PARAM=VALUE adds the interleaved `ours-ab` arm
    # (bench/speed/forest_speed_arm.py --ours-ab), in every mode but stage.
    _ab=""
    [ -n "${MOJOLEARN_SPEED_OURS_AB:-}" ] && _ab="--ours-ab $MOJOLEARN_SPEED_OURS_AB"
    echo "speed $_set $_lane $_ds $_rows rounds=$_rounds mode=$_mode extra='$_extra' $(date -u +%H:%M:%S)"
    _smi=""
    if [ "${MOJOLEARN_SPEED_SMI_SAMPLE:-0}" = 1 ] && command -v rocm-smi > /dev/null 2>&1; then
        ( while :; do echo "t $(date -u +%T)"; rocm-smi --showuse --showmemuse 2>/dev/null | grep -i 'GPU use\|VRAM'; sleep 5; done ) > "$_log.smi" 2>&1 &
        _smi=$!
    fi
    case "$_mode" in
        stage)
            MOJOLEARN_STAGE_TIMES=1 MOJOLEARN_SPEED_ROUNDS=1 timeout -k 30 1800 $PY -u bench/speed/forest_speed_arm.py \
                --lane "$_lane" --dataset "$_ds" --rows "$_rows" --ours-only > "$_log" 2>&1 ;;
        ours)
            MOJOLEARN_SPEED_ROUNDS="$_rounds" timeout -k 30 3600 $PY -u bench/speed/forest_speed_arm.py \
                --lane "$_lane" --dataset "$_ds" --rows "$_rows" --ours-only $_ab > "$_log" 2>&1 ;;
        *)
            # shellcheck disable=SC2086  # $_extra and $_ab are word-split on purpose
            MOJOLEARN_SPEED_ROUNDS="$_rounds" timeout -k 30 3600 $PY -u bench/speed/forest_speed_arm.py \
                --lane "$_lane" --dataset "$_ds" --rows "$_rows" $_extra $_ab > "$_log" 2>&1 ;;
    esac
    _rc=$?
    [ -n "$_smi" ] && kill "$_smi" 2>/dev/null
    echo "speed_exit $_set.$_lane.$_ds.r$_rows.$_mode${MOJOLEARN_SPEED_TAG:+.$MOJOLEARN_SPEED_TAG}=$_rc $(date -u +%H:%M:%S)" | tee -a "$OUT/ab.txt"
    grep -E '^FSPEED(-HEADER|-ACC|-REFUSED)? ' "$_log" | grep -v WARMUP | head -40
}

cmd_rfgate() {
    _set="$1"; shift
    echo "rf_perf_candidates_check $_set defines='$*' $(date -u +%H:%M:%S)"
    # shellcheck disable=SC2086
    timeout -k 30 1500 pixi run mojo run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 $* \
        ensemble/checks/rf_perf_candidates_check.mojo > "$LOGS/rfgate.$_set.log" 2>&1
    echo "rfgate_exit $_set=$? $(date -u +%H:%M:%S)" | tee -a "$OUT/ab.txt"
    tail -5 "$LOGS/rfgate.$_set.log"
}

case "${1:-}" in
    build) shift; cmd_build "$@" ;;
    use) shift; cmd_use "$@" ;;
    ib) shift; cmd_ib "$@" ;;
    diff) shift; cmd_diff "$@" ;;
    speed) shift; cmd_speed "$@" ;;
    rfgate) shift; cmd_rfgate "$@" ;;
    *) sed -n '2,18p' "$0"; exit 2 ;;
esac
