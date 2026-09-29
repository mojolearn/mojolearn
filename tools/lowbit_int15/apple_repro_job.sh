#!/bin/bash
# tools/lowbit_int15/apple_repro_job.sh -- lane/lowbit-int15: THE M2 PRO
# FAILURE, alone. Request 1790653526005 (timing, commit 77598eeb7) stopped at
# llama8b.mlp_down.t512 with "POISON SURVIVED at cell 1723968": a fifteen-bit
# kernel launched and did not write every cell of a 512 x 4096 x 14336
# product. This job runs THAT ROW and nothing else, every arm once and
# nothing timed, three times, with Metal's API validation on, and says what
# each run did.
#
# Metal's validation layer prints, for a dispatch over a pipeline's thread
# limit, "... must be <= N. (kernel threadgroup size limit)" (lane
# linear-apple3's M2 Pro job, tools/linear_apple3/m2.sh). The M2 Pro has no
# Dynamic Caching, so a pipeline's limit falls with its register use, and a
# dispatch over it is DROPPED, not refused.
#
#   MOJOLEARN_LOWBIT_BOX=m2pro bash tools/lowbit_int15/apple_repro_job.sh
set -u
cd "$(dirname "$0")/../.." || exit 9
BOX=${MOJOLEARN_LOWBIT_BOX:-$(hostname -s)}
ROWS=${MOJOLEARN_INT15_REPRO_ROWS:-mlp_down.t512}
OUT="$PWD/bench/results/lowbit_int15/$BOX/repro"
rm -rf "$OUT"
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
{
    echo "box=$BOX rows=$ROWS"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "tree_head=$(git rev-parse HEAD 2>/dev/null)"
    echo "machine=$(sysctl -n machdep.cpu.brand_string 2>/dev/null) macOS $(sw_vers -productVersion 2>/dev/null)"
} > "$OUT/repro.txt"
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/gemm_int15_price_main.mojo -o "$OUT/int15_price" > "$OUT/build.log" 2>&1
echo "build exit=$?" >> "$OUT/repro.txt"
for run in 1 2 3; do
    t0=$(date +%s)
    MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=nslog \
    MOJOLEARN_INT15_PRICE_ONLY="$ROWS" MOJOLEARN_INT15_PRICE_IDENTITY_ONLY=1 \
        pixi run "$OUT/int15_price" > "$OUT/run$run.log" 2>&1
    rc=$?
    {
        echo "run $run exit=$rc $(( $(date +%s) - t0 ))s rows_begun=$(grep -c '^== llama8b' "$OUT/run$run.log") digests=$(grep -c '^DIGEST ' "$OUT/run$run.log") thread_limit_messages=$(grep -c 'threadgroup size limit' "$OUT/run$run.log")"
        grep -h -E 'POISON|PLANS DISAGREE|SCHEDULES DISAGREE|Unhandled' "$OUT/run$run.log" | cut -c1-400
    } >> "$OUT/repro.txt"
done
# Every row of the harness once under the validation layer, for the launches
# this box drops at ANY row.
t0=$(date +%s)
MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=nslog MOJOLEARN_INT15_PRICE_IDENTITY_ONLY=1 \
    MOJOLEARN_INT15_PRICE_ONLY="t1,t8,qkv.t512" pixi run "$OUT/int15_price" > "$OUT/small_rows.log" 2>&1
echo "small rows exit=$? $(( $(date +%s) - t0 ))s thread_limit_messages=$(grep -c 'threadgroup size limit' "$OUT/small_rows.log")" >> "$OUT/repro.txt"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/repro.txt"
cat "$OUT/repro.txt"
echo "== Metal validation: dispatches over a pipeline's thread limit, by message"
cat "$OUT"/run*.log "$OUT/small_rows.log" | grep "threadgroup size limit" | sed 's/^.*\] //' | sort | uniq -c | head -20
echo "== every other line of the validation layer"
cat "$OUT"/run*.log "$OUT/small_rows.log" | grep -i -E "metal|validation|MTL|error" | grep -v "threadgroup size limit" | cut -c1-300 | sort | uniq -c | sort -rn | head -30
for run in 1 2 3; do
    echo "== run $run: the lines of the row"
    grep -E '^(== |INT15|INT15-NOT-RUN)' "$OUT/run$run.log" | cut -c1-260 | tail -40
done
echo "apple_repro_job: box=$BOX"
exit 0
