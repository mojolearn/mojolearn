#!/bin/bash
# Stage split (MOJOLEARN_STAGE_TIMES=1, a TRIAGE run, never a timing) of FAST
# tree fits on the box this runs on (lane apple-fast-trees2). Builds the FAST
# binding once with "<defines>" (skipped when AFT_SKIP_BUILD=1 and a .so is
# there), then runs the board's tree driver once per lane:dataset pair with
# the stage clock on, printing the stage tables and the FSPEED lines.
#
#   bash tools/aft_stage.sh <binding> "<defines>" <lane:dataset> [<lane:dataset> ...]
set -u
bind=$1; defs=$2; shift 2
here=$(cd "$(dirname "$0")/.." && pwd); cd "$here"
py=${AFT_PY:-python3}
[ -z "${AFT_PY:-}" ] && [ -x "$HOME/board-0834/cache/venv/bin/python" ] && py=$HOME/board-0834/cache/venv/bin/python
out=${AFT_OUT:-$HOME/aft-stage/$bind}
mkdir -p "$out"
export GBM_BENCH_DATA=${GBM_BENCH_DATA:-$HOME/datasets/gbm-bench}
echo "AFT-STAGE head=$(git rev-parse --short HEAD) bind=$bind defs='$defs'"
if [ ! -f python/mojolearn/_mojolearn.so ]; then
    MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1 bash bindings/build.sh > "$out/build_base.log" 2>&1
    echo "AFT-BUILD base rc=$?"
fi
if [ "${AFT_SKIP_BUILD:-0}" != 1 ] || [ ! -f "python/mojolearn/_mojolearn_$bind.so" ]; then
    MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_EXTRA_DEFINES="$defs" MOJOLEARN_SKIP_BUILD_GATE=1 \
        bash bindings/build_$bind.sh > "$out/build.log" 2>&1
    rc=$?; echo "AFT-BUILD $bind rc=$rc"
    [ $rc = 0 ] || { grep -m 5 -B 2 -A 8 error "$out/build.log"; exit 1; }
fi
for pair in "$@"; do
    lane=${pair%%:*}; ds=${pair#*:}
    log="$out/stage_${lane}_$ds.log"
    MOJOLEARN_STAGE_TIMES=1 MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SPEED_ROUNDS=1 MOJOLEARN_SPEED_SIZE=shipped \
    MOJOLEARN_SPEED_EXPECTED_VENDOR=metal PYTHONPATH="$here/python${PYTHONPATH:+:$PYTHONPATH}" \
        "$py" -u bench/speed/forest_speed_arm.py --lane "$lane" --dataset "$ds" --ours-only > "$log" 2>&1
    echo "AFT-STAGE-RUN lane=$lane ds=$ds rc=$?"
    grep -E '^FSPEED(-ACC|-HEADER)? |\[stage-times\]|STAGE_TIMES|^  [A-Za-z_.:/0-9-]+	' "$log" | sed "s/^/AFT-ST $lane:$ds /" | head -120
    grep -m 3 -iE 'Traceback|Error' "$log"
done
