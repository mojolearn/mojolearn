#!/bin/bash
# Alternating FAST A/B of one tree binding on the box this runs on (lane
# apple-fast-trees2). Builds the binding twice in FAST mode (arm A with
# defines $A, arm B with defines $B), then times the board's tree driver with
# each arm's .so in place, alternating A B A B ... for $N pairs, one timed
# round per process. Prints every FSPEED / FSPEED-ACC line prefixed with its
# arm and pair, then the medians.
#
#   bash tools/aft_ab.sh <binding> <lane> <dataset> <pairs> "<defines A>" "<defines B>"
#   e.g. bash tools/aft_ab.sh gbdt gbdt-lossguide taxi 3 "" "-D MOJOLEARN_GBDT_LG_EXACT_BATCH"
#
# binding: rf | gbdt | trees | x_trees (bindings/build_<binding>.sh,
# python/mojolearn/_mojolearn_<binding>.so). Arm B's .so is left installed.
# Env: GBM_BENCH_DATA (default ~/datasets/gbm-bench), AFT_PY (default python3),
# AFT_ROWS (driver --rows), AFT_SKIP_BUILD=1 reuses the builds in the out dir.
set -u
bind=$1; lane=$2; ds=$3; pairs=$4; defA=$5; defB=$6
here=$(cd "$(dirname "$0")/.." && pwd); cd "$here"
py=${AFT_PY:-python3}
[ -z "${AFT_PY:-}" ] && [ -x "$HOME/board-0834/cache/venv/bin/python" ] && py=$HOME/board-0834/cache/venv/bin/python
out=${AFT_OUT:-$HOME/aft-ab/$bind}
so=python/mojolearn/_mojolearn_$bind.so
mkdir -p "$out"
export GBM_BENCH_DATA=${GBM_BENCH_DATA:-$HOME/datasets/gbm-bench}
echo "AFT-AB head=$(git rev-parse --short HEAD) bind=$bind lane=$lane ds=$ds pairs=$pairs A='$defA' B='$defB'"

build() {  # $1 arm, $2 defines
    if [ "${AFT_SKIP_BUILD:-0}" = 1 ] && [ -f "$out/$1.so" ]; then return 0; fi
    MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_EXTRA_DEFINES="$2" MOJOLEARN_SKIP_BUILD_GATE=1 \
        bash bindings/build_$bind.sh > "$out/build_$1.log" 2>&1
    rc=$?
    echo "AFT-BUILD arm=$1 rc=$rc"
    [ $rc = 0 ] || { grep -m 5 -B 2 -A 8 error "$out/build_$1.log"; exit 1; }
    cp "$so" "$out/$1.so"
}
install() { cp "$out/$1.so" "$so.aft" && mv -f "$so.aft" "$so"; }
if [ ! -f python/mojolearn/_mojolearn.so ]; then  # the FAST base binding (helpers every estimator imports)
    MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1 bash bindings/build.sh > "$out/build_base.log" 2>&1
    echo "AFT-BUILD base rc=$?"
fi
build A "$defA"
build B "$defB"
rows=""; [ -n "${AFT_ROWS:-}" ] && rows="--rows $AFT_ROWS"
for i in $(seq 1 "$pairs"); do
    for arm in A B; do
        install $arm
        MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SPEED_ROUNDS=1 MOJOLEARN_SPEED_SIZE=shipped \
        MOJOLEARN_SPEED_EXPECTED_VENDOR=metal PYTHONPATH="$here/python${PYTHONPATH:+:$PYTHONPATH}" \
            "$py" -u bench/speed/forest_speed_arm.py --lane "$lane" --dataset "$ds" --ours-only $rows \
            > "$out/run_${arm}_$i.log" 2>&1
        echo "AFT-RUN arm=$arm pair=$i rc=$?"
        grep -E '^FSPEED(-ACC|-REFUSED|-HEADER)? ' "$out/run_${arm}_$i.log" | sed "s/^/AFT arm=$arm pair=$i /"
        grep -m 3 -iE 'Traceback|Error' "$out/run_${arm}_$i.log"
    done
done
"$py" - "$out" "$pairs" <<'EOF'
import re, sys, glob, statistics
out, pairs = sys.argv[1], int(sys.argv[2])
for arm in "AB":
    ms, hs, acc = [], set(), {}
    for i in range(1, pairs + 1):
        try:
            txt = open("%s/run_%s_%d.log" % (out, arm, i)).read()
        except OSError:
            continue
        for m in re.finditer(r"^FSPEED lane=\S+ arm=ours\S* shape=\S+ round=\d+ ms=([\d.]+) hash=(\S+)", txt, re.M):
            ms.append(float(m.group(1))); hs.add(m.group(2))
        for m in re.finditer(r"^FSPEED-ACC lane=\S+ arm=ours\S* metric=(\S+) value=(\S+)", txt, re.M):
            acc.setdefault(m.group(1), []).append(float(m.group(2)))
    print("AFT-MEDIAN arm=%s n=%d median_ms=%s all=%s hashes=%s acc=%s" % (
        arm, len(ms), round(statistics.median(ms), 1) if ms else None,
        [round(x) for x in ms], sorted(hs),
        {k: [round(x, 6) for x in v] for k, v in acc.items()}))
EOF
