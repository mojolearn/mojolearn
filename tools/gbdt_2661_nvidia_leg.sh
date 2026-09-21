#!/bin/sh
# Focused NVIDIA IDENTICAL trial for DEVIATION 2661.  Run this on a RunPod
# box already shipped and R2-staged by tools/trees_leg.sh.  It compares the
# current GBDT binding with the same source compiled with the non-symmetric
# per-group bit-width path enabled.  There are no opponent arms.
#
# Five outer passes reverse the process order on every pass.  Each process
# runs the forest harness's own warm-up plus five measured rounds at 1M rows
# on exactly Taxi and Istella-S, for Depthwise and Lossguide.
set -u

ROOT=/root/mojolearn
OUT=/root/trees_out
RUN="$OUT/gbdt2661-nvidia"
BINS=/root/bins
TIER="$ROOT/python/mojolearn/identical"
LANES=gbdt-depthwise,gbdt-lossguide
OUTER=${MOJOLEARN_2661_OUTER:-5}
ROUNDS=5

die() {
    printf 'GBDT2661 FAIL: %s\n' "$*" >&2
    printf 'failed\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$RUN/status.tsv"
    exit 1
}

step() { # <name> <seconds> <command...>
    _name=$1
    _cap=$2
    shift 2
    _start=$(date +%s)
    timeout -k 30 "$_cap" "$@" > "$RUN/$_name.log" 2>&1
    _rc=$?
    printf '%s\t%s\t%s\n' "$_name" "$_rc" "$(( $(date +%s) - _start ))" >> "$RUN/status.tsv"
    return "$_rc"
}

require_file() {
    [ -s "$1" ] || die "missing or empty file: $1"
}

[ "$(id -u)" = 0 ] || { echo "this body requires root" >&2; exit 2; }
[ "$OUTER" -ge 5 ] 2>/dev/null || { echo "MOJOLEARN_2661_OUTER must be an integer >= 5" >&2; exit 2; }
[ -d "$ROOT" ] || { echo "missing shipped tree $ROOT" >&2; exit 2; }
cd "$ROOT" || exit 9
rm -rf "$RUN"
mkdir -p "$RUN" "$OUT/logs" "$OUT/ib" "$OUT/speed" "$BINS"
rm -f "$OUT/ib/base2661.json" "$OUT/ib/g2661.json" \
    "$OUT/speed/base2661.gbdt-depthwise."*.ours.o*.log \
    "$OUT/speed/base2661.gbdt-lossguide."*.ours.o*.log \
    "$OUT/speed/g2661.gbdt-depthwise."*.ours.o*.log \
    "$OUT/speed/g2661.gbdt-lossguide."*.ours.o*.log
: > "$RUN/status.tsv"

PATH="$HOME/.pixi/bin:$PATH"
export PATH
export GBM_BENCH_DATA=/root/datasets/gbm-bench
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_SKIP_BUILD_GATE=1
export MOJOLEARN_COMPILE_JOBS="${MOJOLEARN_COMPILE_JOBS:-4}"
export MOJOLEARN_BUILD_JOBS="${MOJOLEARN_BUILD_JOBS:-4}"
unset MOJOLEARN_BUILD_EXTRA_DEFINES MOJOLEARN_EXTRA_DEFINES

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$RUN/gpu.txt" 2>&1 || die "nvidia-smi failed"
for _data in \
    "$GBM_BENCH_DATA/taxi/taxi_speed.npz" \
    "$GBM_BENCH_DATA/istella/istella_speed.npz"
do
    require_file "$_data"
done

# trees_leg ships source and stages data.  Install the pinned environment,
# then build a fresh baseline rather than trusting binaries left on the box.
if ! command -v pixi >/dev/null 2>&1; then
    step pixi_get 300 sh -c 'curl -fsSL https://pixi.sh/install.sh | sh' || die "pixi bootstrap failed"
fi
if [ ! -x "$ROOT/.pixi/envs/default/bin/python3" ]; then
    step pixi_install 1500 pixi install || die "pixi install failed"
fi
PY="$ROOT/.pixi/envs/default/bin/python3"
require_file "$PY"
export MOJOLEARN_SPEED_PY="$PY"

rm -rf "$BINS/baseline" "$BINS/base2661" "$BINS/g2661"
rm -f "$TIER"/*.so
step build_base 1500 sh bindings/build.sh || die "baseline base binding build failed"
step build_gbdt_baseline 1500 sh bindings/build_gbdt.sh || die "baseline GBDT binding build failed"
require_file "$TIER/_mojolearn.so"
require_file "$TIER/_mojolearn_gbdt.so"
mkdir -p "$BINS/baseline" "$BINS/base2661"
cp "$TIER/_mojolearn.so" "$TIER/_mojolearn_gbdt.so" "$BINS/baseline/" || die "could not save baseline bindings"
cp "$BINS/baseline/"*.so "$BINS/base2661/" || die "could not make named baseline set"
sha256sum "$BINS/base2661/"*.so > "$RUN/base.sha256" || die "could not hash baseline bindings"

step build_g2661 1500 sh tools/trees_identical_ab.sh build g2661 gbdt \
    -D MOJOLEARN_2661_NONSYM_GROUP_WIDTH=1 || die "2661 GBDT binding build failed"
require_file "$BINS/g2661/_mojolearn_gbdt.so"
sha256sum "$BINS/g2661/"*.so > "$RUN/g2661.sha256" || die "could not hash candidate bindings"

# Full identity_break columns for both affected lanes.  The helper's `use`
# supplies each named binary set; identity_break itself is called directly so
# --fail-on-refused and the CUDA backend requirement are hard gates.
_gpu_slug=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 \
    | tr '[:upper:] ' '[:lower:]-' | sed 's/[^a-z0-9_.-]/-/g; s/--*/-/g; s/^-//; s/-$//')
VENDOR="nvidia-${_gpu_slug:-cuda}"
for _set in base2661 g2661; do
    step "use_$_set" 60 sh tools/trees_identical_ab.sh use "$_set" || die "could not activate $_set"
    step "identity_$_set" 1800 env PYTHONPATH="$ROOT/python" MOJOLEARN_COMMIT="$(cat SHIPPED_COMMIT.txt 2>/dev/null || true)" \
        "$PY" -u tools/identity_break.py --lanes "$LANES" --repeats 2 \
        --require-backend cuda --fail-on-refused --vendor "$VENDOR" \
        --json "$OUT/ib/$_set.json" || die "identity_break failed for $_set"
    require_file "$OUT/ib/$_set.json"
done
step identity_diff 300 env PYTHONPATH="$ROOT/python" "$PY" -u tools/identity_break.py \
    --diff "$OUT/ib/base2661.json" "$OUT/ib/g2661.json" \
    --lanes "$LANES" --require-columns 2 || die "baseline/candidate identity diff failed"

# The narrow-width regression matrix is an independent exact-reference gate.
for _set in base2661 g2661; do
    step "use_subbyte_$_set" 60 sh tools/trees_identical_ab.sh use "$_set" || die "could not activate $_set for sub-byte gate"
    step "subbyte_$_set" 900 env PYTHONPATH="$ROOT/python" MOJOLEARN_COMMIT="$(cat SHIPPED_COMMIT.txt 2>/dev/null || true)" \
        "$PY" -u checks/gbdt_sub_byte_identity_check.py --lanes "$LANES" \
        --repeats 2 --json "$RUN/subbyte.$_set.json" || die "sub-byte identity gate failed for $_set"
    require_file "$RUN/subbyte.$_set.json"
done

# Five or more outer A/B passes.  Odd passes run baseline first; even passes
# run candidate first.  A unique tag prevents any visit from overwriting an
# earlier harness log.  `ours` mode excludes every opponent.
_outer=1
while [ "$_outer" -le "$OUTER" ]; do
    if [ $(( _outer % 2 )) -eq 1 ]; then
        _sets="base2661 g2661"
    else
        _sets="g2661 base2661"
    fi
    for _lane in gbdt-depthwise gbdt-lossguide; do
        for _ds in taxi istella; do
            for _set in $_sets; do
                _tag="o$_outer"
                _key="$_set.$_lane.$_ds.r1000000.ours.$_tag"
                step "speed.$_key" 3900 env MOJOLEARN_SPEED_TAG="$_tag" \
                    MOJOLEARN_SPEED_PY="$PY" sh tools/trees_identical_ab.sh \
                    speed "$_set" "$_lane" "$_ds" 1000000 "$ROUNDS" ours \
                    || die "timing helper failed for $_key"
                _log="$OUT/speed/$_key.log"
                require_file "$_log"
                grep "^speed_exit $_key=" "$OUT/ab.txt" | tail -1 | grep -q "^speed_exit $_key=0 " \
                    || die "timing child reported failure for $_key"
                _n=$(awk -v lane="$_lane" '$1=="FSPEED" && $0 ~ ("lane=" lane " ") && $0 ~ /arm=ours / {n++} END {print n+0}' "$_log")
                [ "$_n" -eq "$ROUNDS" ] || die "$_key has $_n measured FSPEED rows, expected $ROUNDS"
            done
        done
    done
    _outer=$(( _outer + 1 ))
done

# Full prediction hashes must be stable within and between the two builds for
# every large-data cell.  Then flip_verdict prices the candidate over all 25
# measurements per side and checks the reported quality metrics.  A slower
# candidate is a valid completed experiment; a parse error or quality loss is
# a failed run.
for _lane in gbdt-depthwise gbdt-lossguide; do
    for _ds in taxi istella; do
        _hashes=$(awk '$1=="FSPEED" {for (i=1;i<=NF;i++) if ($i ~ /^hash=/) print substr($i,6)}' \
            "$OUT/speed/base2661.$_lane.$_ds.r1000000.ours.o"*.log \
            "$OUT/speed/g2661.$_lane.$_ds.r1000000.ours.o"*.log | sort -u)
        _nh=$(printf '%s\n' "$_hashes" | awk 'NF {n++} END {print n+0}')
        [ "$_nh" -eq 1 ] || die "output hashes moved for $_lane/$_ds: $_hashes"
        [ "$_hashes" != "-" ] || die "output hash was not recorded for $_lane/$_ds"
    done

    # Shell glob expansion intentionally supplies every outer-pass log to the
    # four nargs+ options.
    _vstart=$(date +%s)
    "$PY" tools/flip_verdict.py --lane "$_lane" --arm ours --rows 1000000 \
        --taxi-before "$OUT/speed/base2661.$_lane.taxi.r1000000.ours.o"*.log \
        --taxi-after "$OUT/speed/g2661.$_lane.taxi.r1000000.ours.o"*.log \
        --istella-before "$OUT/speed/base2661.$_lane.istella.r1000000.ours.o"*.log \
        --istella-after "$OUT/speed/g2661.$_lane.istella.r1000000.ours.o"*.log \
        > "$RUN/verdict.$_lane.txt" 2>&1
    _vrc=$?
    printf 'verdict.%s\t%s\t%s\n' "$_lane" "$_vrc" "$(( $(date +%s) - _vstart ))" >> "$RUN/status.tsv"
    [ "$_vrc" -ne 2 ] || die "flip_verdict could not parse $_lane"
    if grep -Eq 'WORSE|reason=quality:|missing:' "$RUN/verdict.$_lane.txt"; then
        die "quality gate failed for $_lane"
    fi
    tail -1 "$RUN/verdict.$_lane.txt" >> "$RUN/verdicts.txt"
done

printf 'complete\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN/status.tsv"
: > "$RUN/complete"
cat "$RUN/verdicts.txt"
