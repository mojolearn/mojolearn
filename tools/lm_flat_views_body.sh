#!/bin/sh
# Default-off Byte-LM parameter/gradient sub-buffer qualification body.
# The provider runner must stage exactly the Taxi and Istella-S R2 objects.
set -u
ROOT=${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_LM_FLAT_VIEW_OUT:-/root/gemm_leg_out/lm-flat-views}
PATH="$HOME/.pixi/bin:$PATH"
export PATH PYTHONPATH="$ROOT/python:$ROOT" MOJOLEARN_NUMERIC_MODE=identical
cd "$ROOT" || exit 9
mkdir -p "$OUT/bin"
STATUS="$OUT/status.txt"
if [ -n "${MOJOLEARN_COMMIT:-}" ]; then COMMIT=$MOJOLEARN_COMMIT
elif [ -f SHIPPED_COMMIT.txt ]; then COMMIT=$(cat SHIPPED_COMMIT.txt)
elif [ -f MOJOLEARN_COMMIT ]; then COMMIT=$(cat MOJOLEARN_COMMIT)
else COMMIT=$(git rev-parse HEAD) || exit 9
fi
if [ -d .git ]; then
    [ -z "$(git status --porcelain --untracked-files=no)" ] || {
        echo "refusing dirty tracked source tree" >&2; exit 9;
    }
fi
TAXI="$HOME/datasets/gbm-bench/taxi/taxi_speed.npz"
ISTELLA="$HOME/datasets/gbm-bench/istella/istella_speed.npz"
TAXI_KEY=gbm-bench/taxi/taxi_speed.npz
ISTELLA_KEY=gbm-bench/istella/istella_speed.npz
SHAPE=${MOJOLEARN_LM_FLAT_VIEW_SHAPE:-"1 2048 768 12 12 64 2048 12 50257"}
WITNESS=${MOJOLEARN_LM_FLAT_VIEW_WITNESS_STEPS:-3}
WARMUP=${MOJOLEARN_LM_FLAT_VIEW_WARMUP:-3}
SAMPLES=${MOJOLEARN_LM_FLAT_VIEW_SAMPLES:-9}
PROCESSES=${MOJOLEARN_LM_FLAT_VIEW_PROCESSES:-3}
BIND=python/mojolearn/identical/_mojolearn_byte_lm.so
[ "$PROCESSES" -eq 3 ] || { echo "gate requires exactly 3 processes" >&2; exit 2; }
if [ "$WITNESS" -ne 3 ] || [ "$WARMUP" -ne 3 ] || [ "$SAMPLES" -ne 9 ]; then
    echo "gate requires witness=3 warmup=3 samples=9" >&2; exit 2
fi
case "${MOJOLEARN_TARGET_COLUMN:-}" in nvidia|amd) ;; *)
    echo "MOJOLEARN_TARGET_COLUMN must be nvidia or amd" >&2; exit 2;; esac

echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$COMMIT" > "$STATUS"
for pair in "$TAXI_KEY:$TAXI" "$ISTELLA_KEY:$ISTELLA"; do
    key=${pair%%:*}; path=${pair#*:}
    sh tools/dataset_store.sh verify "$key" "$path" >> "$STATUS" 2>&1 || exit 2
done
echo "datasets=R2:$TAXI_KEY,R2:$ISTELLA_KEY shape=$SHAPE" >> "$STATUS"

# Build the common base binding once. Byte-LM arms go to distinct fresh dirs.
rm -f python/mojolearn/identical/_mojolearn.so "$BIND"
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/build-base.log" 2>&1 || exit 3

build_arm() { # name defines
    name=$1; defs=$2
    mkdir -p "$OUT/bin/$name"
    t0=$(date +%s)
    MOJOLEARN_BYTE_LM_OUTDIR="$OUT/bin/$name" \
      MOJOLEARN_BUILD_EXTRA_DEFINES="$defs" MOJOLEARN_COMPILE_JOBS=2 \
      sh bindings/build_byte_lm.sh > "$OUT/build-$name.log" 2>&1
    rc=$?
    echo "build $name exit=$rc secs=$(( $(date +%s) - t0 )) defines=$defs" >> "$STATUS"
    [ "$rc" -eq 0 ] || exit "$rc"
}
build_arm off ""
build_arm param "-D MOJOLEARN_BYTE_LM_PARAM_VIEWS=1"
build_arm grad "-D MOJOLEARN_BYTE_LM_GRAD_VIEWS=1"
build_arm both "-D MOJOLEARN_BYTE_LM_PARAM_VIEWS=1 -D MOJOLEARN_BYTE_LM_GRAD_VIEWS=1"
build_arm param_sabotage "-D MOJOLEARN_BYTE_LM_PARAM_VIEWS=1 -D MOJOLEARN_BYTE_LM_GRAD_VIEWS=1 -D MOJOLEARN_BYTE_LM_FLAT_VIEW_SABOTAGE=1"
build_arm grad_sabotage "-D MOJOLEARN_BYTE_LM_PARAM_VIEWS=1 -D MOJOLEARN_BYTE_LM_GRAD_VIEWS=1 -D MOJOLEARN_BYTE_LM_GRAD_VIEW_SABOTAGE=1"
sha256sum "$OUT"/bin/*/_mojolearn_byte_lm.so > "$OUT/binding-sha256.txt"

install_arm() { cp "$OUT/bin/$1/_mojolearn_byte_lm.so" "$BIND"; }
probe() { # dataset-label path key result-label binary-arm process position witness warmup samples
    ds=$1; path=$2; key=$3; label=$4; binary=$5; process=$6
    position=$7; witness=$8; warm=$9; shift 9; samples=$1
    install_arm "$binary"
    if [ "$process" = sabotage ]; then
        dest="$OUT/$ds/$label"
    else
        dest="$OUT/$ds/$label/process$process"
    fi
    mkdir -p "$dest"
    t0=$(date +%s)
    # shellcheck disable=SC2086 # nine shape integers are intentional argv
    pixi run python tools/lm_flat_views_probe.py --dataset "$path" \
      --dataset-key "$key" --out "$dest/result.json" --shape $SHAPE \
      --commit "$COMMIT" --process "$process" --position "$position" \
      --target-column "$MOJOLEARN_TARGET_COLUMN" \
      --witness-steps "$witness" --warmup "$warm" --samples "$samples" \
      > "$dest/probe.log" 2>&1
    rc=$?
    echo "probe $ds/$label/$process binary=$binary exit=$rc secs=$(( $(date +%s) - t0 ))" >> "$STATUS"
    [ "$rc" -eq 0 ] || exit "$rc"
}

for row in "taxi:$TAXI:$TAXI_KEY" "istella:$ISTELLA:$ISTELLA_KEY"; do
    ds=${row%%:*}; rest=${row#*:}; path=${rest%%:*}; key=${rest#*:}
    process=0
    while [ "$process" -lt "$PROCESSES" ]; do
        # Interleave arms so the median of three process medians sees the
        # same box-temperature window. Stability remains diagnostic only.
        case "$process" in
            0) order="off param grad both" ;;
            1) order="both grad param off" ;;
            2) order="param off both grad" ;;
            *) order="off both param grad" ;;
        esac
        position=0
        for arm in $order; do
            probe "$ds" "$path" "$key" "$arm" "$arm" "$process" \
              "$position" "$WITNESS" "$WARMUP" "$SAMPLES"
            position=$((position + 1))
        done
        process=$((process + 1))
    done
    probe "$ds" "$path" "$key" param_sabotage param_sabotage 0 0 2 0 0
    probe "$ds" "$path" "$key" grad_sabotage grad_sabotage 0 0 2 0 0
done

install_arm both
pixi run python tools/byte_lm_session_check.py --run > "$OUT/session-check.log" 2>&1 || exit 4
pixi run python tools/lm_flat_views_pooled_check.py --dataset "$TAXI" \
  --out "$OUT/pooled-check.json" > "$OUT/pooled-check.log" 2>&1 || exit 5

if [ "${MOJOLEARN_TARGET_COLUMN:-}" = amd ]; then
    build_arm both_fault "-D MOJOLEARN_BYTE_LM_PARAM_VIEWS=1 -D MOJOLEARN_BYTE_LM_GRAD_VIEWS=1 -D MOJOLEARN_BYTE_LM_FAULT_INJECT=1"
    install_arm both_fault
    pixi run python tools/byte_lm_session_check.py --run > "$OUT/native-fault-check.log" 2>&1 || exit 6
fi

pixi run python tools/lm_flat_views_compare.py --root "$OUT" \
  --out "$OUT/verdict.json" > "$OUT/verdict.log" 2>&1
rc=$?
cat "$OUT/verdict.log" >> "$STATUS"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) verdict_exit=$rc" >> "$STATUS"
exit "$rc"
