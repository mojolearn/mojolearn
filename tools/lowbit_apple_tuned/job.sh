#!/bin/bash
# tools/lowbit_apple_tuned/job.sh -- lane/lowbit-apple-tuned: THE GATE, THEN
# THE CLOCK, of the tuned Apple float-unit plans of int15i64.v1, on ONE Mac,
# as one steward job. It follows tools/lowbit_int15/gate_job.sh and
# price_job.sh (lane/lowbit-int15): every phase has its own exit code and
# EXPECTED verdict, a later phase runs even when an earlier one failed (a red
# phase is a finding), and the exit is non-zero when any expectation did not
# hold.
#
#   bash tools/lowbit_apple_tuned/job.sh <box name>        (m3ultra, m2pro)
#
#   gate                 gemm/checks/gemm_int15_apple_tuned_check.mojo
#                        EXPECTED exit 0: every variant equals the host
#                        oracle, the flat kernel and the simulation's vectors.
#   gate-chunk-sabotage  -D MOJOLEARN_INT15_APPLE_TUNED_CHUNK_SABOTAGE=1
#                        THE ARM THAT REMOVES THE CHUNK BOUNDARY. EXPECTED
#                        non-zero, naming the planted worst cases and the
#                        chunk boundaries, and leaving the shapes inside one
#                        chunk passing.
#   gate-value-sabotage  -D MOJOLEARN_LOWBIT_SABOTAGE=1
#                        THE ARM THAT FLIPS A VALUE. EXPECTED non-zero,
#                        naming every device gate.
#   build                ONE binary of the harness. Nothing is timed while
#                        it builds.
#   warm                 the binary, identity only: every arm of every row
#                        once, untimed (never time a first run after a build).
#   price                the SAME binary, timed. EXPECTED exit 0.
#   price-sabotage       the value arm, identity only, the qkv rows: every
#                        tuned digest must DIFFER from the clean run's.
#
# ISOLATION. The steps are in sequence and never overlap; the steward runs
# one job at a time on a Mac.
#
# ENVIRONMENT: MOJOLEARN_TUNED_PRICE_ONLY, _VARIANTS, _REPEATS, _SLICE_MACS,
# _FLAT pass through to the harness. MOJOLEARN_TUNED_JOB_PHASES (default
# "gate price") picks the phases.
set -u
cd "$(dirname "$0")/../.." || exit 9
[ $# -ge 1 ] || { echo "job.sh <box>" >&2; exit 2; }
BOX=$1
PHASES=${MOJOLEARN_TUNED_JOB_PHASES:-gate price}
OUT="$PWD/bench/results/lowbit_apple_tuned/$BOX"
rm -rf "$OUT"
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
if [ "$(uname -s)" != Darwin ]; then
    echo "job.sh: not a Mac; NOTHING RUN, which is not a pass"
    exit 9
fi
red=0
{
    echo "profile=mojolearn.identical.gemm.int15i64.v1 (the tuned Apple float-unit plans)"
    echo "box=$BOX phases=$PHASES"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "tree_head=$(git rev-parse HEAD 2>/dev/null) dirty_files=$(git status --porcelain 2>/dev/null | grep -c .)"
    echo "machine=$(sysctl -n machdep.cpu.brand_string 2>/dev/null) macOS $(sw_vers -productVersion 2>/dev/null)"
    echo "only=${MOJOLEARN_TUNED_PRICE_ONLY:-all} variants=${MOJOLEARN_TUNED_PRICE_VARIANTS:-all} repeats=${MOJOLEARN_TUNED_PRICE_REPEATS:-default} slice=${MOJOLEARN_TUNED_PRICE_SLICE_MACS:-default} flat=${MOJOLEARN_TUNED_PRICE_FLAT:-0}"
} > "$OUT/job.txt"

CHECK=gemm/checks/gemm_int15_apple_tuned_check.mojo
BENCH=bench/gemm_int15_apple_tuned_price_main.mojo

run() {
    # run <name> <pass|fail> <command...>: an arm HELD only when the program
    # ran to its verdict line and failed there; a build that does not compile
    # also exits non-zero, and that is not a sabotage seen failing.
    _name=$1
    _want=$2
    shift 2
    _t0=$(date +%s)
    "$@" > "$OUT/$_name.log" 2>&1
    _code=$?
    _held=no
    if [ "$_want" = pass ] && [ "$_code" -eq 0 ]; then _held=yes; fi
    if [ "$_want" = fail ] && [ "$_code" -ne 0 ] && grep -q -E '^== [0-9]+ gates, [1-9][0-9]* failed ==$' "$OUT/$_name.log"; then _held=yes; fi
    printf '%s\t%s\t%s\texpected=%s\theld=%s\n' "$_name" "$_code" "$(( $(date +%s) - _t0 ))s" "$_want" "$_held" >> "$OUT/status.tsv"
    [ "$_held" = yes ] || red=1
    return "$_code"
}
must_name() {
    if ! grep -q "GATE FAILED: $2" "$OUT/$1.log" 2>/dev/null; then
        echo "reach: $1 did not fail $2" >> "$OUT/job.txt"
        red=1
    else
        echo "reach: $1 failed $2, as it must" >> "$OUT/job.txt"
    fi
}
must_pass() {
    if ! grep -q "^ok $2\$" "$OUT/$1.log" 2>/dev/null; then
        echo "reach: $1 did not leave $2 passing" >> "$OUT/job.txt"
        red=1
    else
        echo "reach: $1 left $2 passing, as it must" >> "$OUT/job.txt"
    fi
}
show() {
    # Every line of a gate log that is not a digest or a per-case ok.
    echo "== $1 (every line that is not a digest or a per-case ok; last 80)"
    grep -v -E '^   (DIGEST|ok) |warning:|^ *\^|^ *[0-9]* *\||^$|deprecated' "$OUT/$1.log" 2>/dev/null | cut -c1-1200 | tail -80
    if grep -q -E 'error:' "$OUT/$1.log" 2>/dev/null; then
        echo "== $1: the compiler's errors"
        grep -A6 -E 'error:' "$OUT/$1.log" | cut -c1-400 | head -120
    fi
}

for phase in $PHASES; do
    echo "######## phase $phase on $BOX, started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    case "$phase" in
    gate)
        run gate pass pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . "$CHECK"
        run gate-chunk-sabotage fail pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_INT15_APPLE_TUNED_CHUNK_SABOTAGE=1 -I . "$CHECK"
        must_name gate-chunk-sabotage check_tuned_planted_worst_cases
        must_name gate-chunk-sabotage check_tuned_chunk_boundaries
        must_pass gate-chunk-sabotage check_tuned_shapes_inside_one_chunk
        run gate-value-sabotage fail pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_LOWBIT_SABOTAGE=1 -I . "$CHECK"
        must_name gate-value-sabotage check_tuned_shapes_inside_one_chunk
        must_name gate-value-sabotage check_tuned_shapes_across_chunks
        must_name gate-value-sabotage check_tuned_planted_worst_cases
        must_name gate-value-sabotage check_tuned_chunk_boundaries
        must_name gate-value-sabotage check_tuned_launch_in_slices
        must_name gate-value-sabotage check_tuned_simulation_vectors
        grep -h "^   DIGEST " "$OUT/gate.log" 2>/dev/null | sed 's/^   //' > "$OUT/gate_digests.tsv"
        echo "gate digests=$(grep -c . "$OUT/gate_digests.tsv")" >> "$OUT/job.txt"
        cat "$OUT/status.tsv"
        show gate
        show gate-chunk-sabotage
        show gate-value-sabotage
        ;;
    price)
        t0=$(date +%s)
        pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . "$BENCH" -o "$OUT/tuned_price" > "$OUT/build.log" 2>&1
        rc=$?
        echo "build exit=$rc $(( $(date +%s) - t0 ))s" >> "$OUT/job.txt"
        if [ "$rc" -ne 0 ]; then
            red=1
            grep -A6 -E 'error:' "$OUT/build.log" | cut -c1-400 | head -120
        else
            t0=$(date +%s)
            MOJOLEARN_TUNED_PRICE_IDENTITY_ONLY=1 pixi run "$OUT/tuned_price" > "$OUT/warm.log" 2>&1
            rc=$?
            echo "warm exit=$rc $(( $(date +%s) - t0 ))s (untimed, every arm once)" >> "$OUT/job.txt"
            [ "$rc" -eq 0 ] || red=1
            t0=$(date +%s)
            pixi run "$OUT/tuned_price" > "$OUT/price.log" 2>&1
            rc=$?
            echo "price exit=$rc $(( $(date +%s) - t0 ))s" >> "$OUT/job.txt"
            [ "$rc" -eq 0 ] || red=1
            grep -E '^DIGEST ' "$OUT/price.log" > "$OUT/price_digests.tsv"
            grep -E '^DIGEST ' "$OUT/warm.log" > "$OUT/warm_digests.tsv"
            if [ -s "$OUT/price_digests.tsv" ] && cmp -s "$OUT/price_digests.tsv" "$OUT/warm_digests.tsv"; then
                echo "warm-and-timed: the two runs printed the same digests ($(grep -c . "$OUT/price_digests.tsv"))" >> "$OUT/job.txt"
            else
                echo "warm-and-timed: the two runs' digests DIFFER, or a run printed none" >> "$OUT/job.txt"
                red=1
            fi
            MOJOLEARN_TUNED_PRICE_ONLY=${MOJOLEARN_TUNED_PRICE_SABOTAGE_ONLY:-qkv} MOJOLEARN_TUNED_PRICE_IDENTITY_ONLY=1 MOJOLEARN_TUNED_PRICE_FLAT=0 \
                pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_LOWBIT_SABOTAGE=1 -I . "$BENCH" > "$OUT/price_sabotage.log" 2>&1
            echo "price-sabotage exit=$? (identity only)" >> "$OUT/job.txt"
            grep -E '^DIGEST price-.* (tuned|inference\.tuned|training\.tuned)\.' "$OUT/price_sabotage.log" > "$OUT/price_sabotage_digests.tsv"
            grep -E '^DIGEST price-.* (tuned|inference\.tuned|training\.tuned)\.' "$OUT/price.log" > "$OUT/price_tuned_digests.tsv"
            python3 tools/lowbit_int15/digests.py --expect-disagree clean="$OUT/price_tuned_digests.tsv" sabotage="$OUT/price_sabotage_digests.tsv" > "$OUT/price_sabotage_verdict.txt" 2>&1
            rc=$?
            echo "price-sabotage-comparison exit=$rc (0: every sabotaged tuned digest differed from the clean one)" >> "$OUT/job.txt"
            [ "$rc" -eq 0 ] || red=1
        fi
        echo "== price.log (every line that is not a digest)"
        grep -v -E '^DIGEST |mbind|warning:|deprecated' "$OUT/price.log" 2>/dev/null | cut -c1-400
        echo "== warm.log (failures and refusals)"
        grep -E 'TUNED-FAILED|PLANS DISAGREE|FAILED at|Unhandled|Error' "$OUT/warm.log" 2>/dev/null | cut -c1-400 | head -40
        echo "== price sabotage verdict"
        cat "$OUT/price_sabotage_verdict.txt" 2>/dev/null
        echo "== price digests (the timed run)"
        cat "$OUT/price_digests.tsv" 2>/dev/null
        ;;
    *)
        echo "job.sh: unknown phase $phase" >&2
        red=1
        ;;
    esac
    echo "######## phase $phase on $BOX finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done
if [ -f "$OUT/gate_digests.tsv" ]; then
    echo "== gate digests (the clean run)"
    cat "$OUT/gate_digests.tsv"
fi
{
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ "$red" -eq 0 ]; then
        echo "verdict=GREEN every expected outcome held"
    else
        echo "verdict=RED an expected outcome did not hold"
    fi
} >> "$OUT/job.txt"
echo "== job"
[ -f "$OUT/status.tsv" ] && cat "$OUT/status.tsv"
cat "$OUT/job.txt"
echo "lowbit_apple_tuned job: box=$BOX red=$red"
exit "$red"
