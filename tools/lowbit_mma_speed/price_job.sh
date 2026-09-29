#!/bin/bash
# tools/lowbit_mma_speed/price_job.sh -- lane/lowbit-mma-speed, the timing
# harness (bench/gemm_lowbit_price_main.mojo, lane/lowbit-units') with this
# lane's arms, on one box, ISOLATED the way the brief asks:
#
#   build    the binary is BUILT FIRST, and nothing is timed while it builds.
#   cold     the binary's first run. A first run after a build is cold (the
#            device compiles and caches every kernel); ITS TIMES ARE NEVER
#            READ. Its digests are kept and must equal the run of record's.
#   price    the second run, THE RUN OF RECORD.
#   price-sabotage  -D MOJOLEARN_LOWBIT_PRICE_SABOTAGE=1, one timed call per
#            arm, the qkv rows (or the clean run's own rows): its digests
#            must DIFFER from the clean run's at every arm
#            (tools/lowbit_units/table.py --expect-disagree).
#
#   bash tools/lowbit_mma_speed/price_job.sh <name>
#
# <name> files the run under <results>/<box>/price_<name>/. The arms, the
# rows, the repeats and the mac budget come from the environment
# (MOJOLEARN_LOWBIT_PRICE_ARMS, _ONLY, _REPEATS, _MAC_BUDGET); THE SAME
# BUDGET MUST RUN ON EVERY BOX whose digests are compared.
# MOJOLEARN_LOWBIT_PRICE_IDENTITY_ONLY=1 runs every arm once and times
# nothing. MOJOLEARN_LOWBIT_PRICE_MAIN names the harness built (the default
# is lane/lowbit-units'; lane/lowbit-amd-tuned's
# bench/gemm_lowbit_amd_price_main.mojo holds every arm of it and the AMD
# plans).
set -u
cd "$(dirname "$0")/../.." || exit 9
NAME=${1:-}
[ -n "$NAME" ] || { echo "price_job.sh <name>" >&2; exit 2; }
BOX=${MOJOLEARN_LOWBIT_BOX:-$(hostname -s)}
MAIN=${MOJOLEARN_LOWBIT_PRICE_MAIN:-bench/gemm_lowbit_price_main.mojo}
OUT="$PWD/${MOJOLEARN_LOWBIT_RESULTS:-bench/results/lowbit_mma_speed}/$BOX/price_$NAME"
rm -rf "$OUT"
mkdir -p "$OUT"
# The binaries live outside the tree, so a build leaves no file in it.
BIN_DIR="${MOJOLEARN_LOWBIT_BIN_DIR:-${TMPDIR:-/tmp}/lowbit-mma-speed-bin}"
mkdir -p "$BIN_DIR"
export PATH="$HOME/.pixi/bin:$PATH"
. tools/lowbit_mma_speed/box_env.sh
red=0
{
    echo "box=$BOX name=$NAME harness=$MAIN"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "tree_head=$(git rev-parse HEAD 2>/dev/null) dirty_files=$(git status --porcelain 2>/dev/null | grep -c .)"
    box_describe
    echo "arms=${MOJOLEARN_LOWBIT_PRICE_ARMS:-all} only=${MOJOLEARN_LOWBIT_PRICE_ONLY:-all} repeats=${MOJOLEARN_LOWBIT_PRICE_REPEATS:-default} mac_budget=${MOJOLEARN_LOWBIT_PRICE_MAC_BUDGET:-default} identity_only=${MOJOLEARN_LOWBIT_PRICE_IDENTITY_ONLY:-0}"
} > "$OUT/run.txt"

pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . "$MAIN" \
    -o "$BIN_DIR/price_$NAME" > "$OUT/build.log" 2>&1
rc=$?
echo "build exit=$rc" >> "$OUT/run.txt"
if [ "$rc" -ne 0 ]; then
    cat "$OUT/run.txt"
    echo "== build.log (tail)"
    tail -60 "$OUT/build.log"
    echo "price_job: box=$BOX name=$NAME red=1 (the build failed; nothing was timed)"
    exit 1
fi
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_LOWBIT_PRICE_SABOTAGE=1 -I . \
    "$MAIN" -o "$BIN_DIR/price_${NAME}_sabotage" > "$OUT/build_sabotage.log" 2>&1
echo "build-sabotage exit=$?" >> "$OUT/run.txt"

echo "cold started=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/run.txt"
pixi run "$BIN_DIR/price_$NAME" > "$OUT/cold.log" 2>&1
rc=$?
echo "cold exit=$rc (the first run after the build: its times are never read)" >> "$OUT/run.txt"
[ "$rc" -eq 0 ] || red=1
grep -E '^LOWBIT(-NOT-RUN)? ' "$OUT/cold.log" > "$OUT/lowbit_cold.tsv"

echo "price started=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/run.txt"
pixi run "$BIN_DIR/price_$NAME" > "$OUT/price.log" 2>&1
rc=$?
echo "price exit=$rc (the run of record)" >> "$OUT/run.txt"
[ "$rc" -eq 0 ] || red=1
grep -E '^LOWBIT(-NOT-RUN)? ' "$OUT/price.log" > "$OUT/lowbit.tsv"

# The two runs of one binary on one box: the same digest at every arm and
# shape, or the run is not deterministic and no time of it is read.
python3 tools/lowbit_mma_speed/table.py --same-digests "$OUT/lowbit_cold.tsv" "$OUT/lowbit.tsv" > "$OUT/cold_vs_record.txt" 2>&1
rc=$?
echo "cold-vs-record exit=$rc (0: the two runs' digests agree at every arm and shape)" >> "$OUT/run.txt"
[ "$rc" -eq 0 ] || red=1

if [ -x "$BIN_DIR/price_${NAME}_sabotage" ]; then
    MOJOLEARN_LOWBIT_PRICE_ONLY=${MOJOLEARN_LOWBIT_PRICE_ONLY:-qkv} MOJOLEARN_LOWBIT_PRICE_REPEATS=1 \
        pixi run "$BIN_DIR/price_${NAME}_sabotage" > "$OUT/price_sabotage.log" 2>&1
    echo "price-sabotage exit=$? (the in-run plan comparison may raise; the verdict is the digest comparison below)" >> "$OUT/run.txt"
    grep -E '^LOWBIT ' "$OUT/price_sabotage.log" > "$OUT/lowbit_sabotage.tsv"
    python3 tools/lowbit_units/table.py --expect-disagree "$OUT/lowbit.tsv" "$OUT/lowbit_sabotage.tsv" > "$OUT/sabotage_verdict.txt" 2>&1
    rc=$?
    echo "sabotage-comparison exit=$rc (0: every sabotaged digest differed from the clean one)" >> "$OUT/run.txt"
    [ "$rc" -eq 0 ] || red=1
else
    echo "price-sabotage NOT RUN: its build failed (build_sabotage.log)" >> "$OUT/run.txt"
    red=1
fi
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/run.txt"

cat "$OUT/run.txt"
echo "== price.log (every line that is not a PRICE or device-bits line)"
grep -v -E '^(PRICE|   device-bits) ' "$OUT/price.log"
if [ -n "${MOJOLEARN_LOWBIT_PRICE_PRINT_COLD:-}" ]; then
    # A second set of times from the same binary, for a box whose job is
    # its only run: the cold run's LOWBIT lines, NEVER the run of record.
    echo "== cold.log LOWBIT lines (the first run after the build; not the run of record)"
    sed -e 's/^LOWBIT/COLD-LOWBIT/' "$OUT/lowbit_cold.tsv"
fi
echo "== cold against the run of record"
cat "$OUT/cold_vs_record.txt"
echo "== sabotage verdict"
cat "$OUT/sabotage_verdict.txt" 2>/dev/null
echo "== the lever table"
python3 tools/lowbit_mma_speed/table.py --levers "$BOX=$OUT/lowbit.tsv" 2>&1
echo "price_job: box=$BOX name=$NAME red=$red"
exit "$red"
