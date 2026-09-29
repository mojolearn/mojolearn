#!/bin/bash
# tools/lowbit_int15/price_job.sh -- lane/lowbit-int15: the timing harness
# (bench/gemm_int15_price_main.mojo) on ONE box, then the arm that shows its
# digests can differ.
#
# ISOLATION (the brief, review point 4). This job must be the ONLY thing of
# the lane's on the box while it runs: no build, no `sh`, no second job.
# Inside it the steps are in sequence and never overlap:
#
#   build            ONE binary, `mojo build`. Nothing is timed while it builds.
#   warm             the binary with MOJOLEARN_INT15_PRICE_IDENTITY_ONLY=1:
#                    every arm of every row once, untimed. Every kernel is
#                    compiled and launched before the first timed call (a
#                    first run is cold; on AMD the codegen is stable only
#                    from a warm cache).
#   price            the SAME binary, timed. EXPECTED exit 0.
#   price-sabotage   -D MOJOLEARN_LOWBIT_SABOTAGE=1, identity only, the qkv
#                    rows (or the clean run's own filter): every fifteen-bit
#                    product digest must DIFFER from the clean run's.
#
# Writes bench/results/lowbit_int15/<box>/price/ and prints the INT15 lines,
# the digests and the verdicts, so a steward's stdout carries them home.
# THE SAME BUDGET MUST RUN ON EVERY BOX whose digests are compared.
set -u
cd "$(dirname "$0")/../.." || exit 9
BOX=${MOJOLEARN_LOWBIT_BOX:-$(hostname -s)}
OUT="$PWD/bench/results/lowbit_int15/$BOX/${MOJOLEARN_INT15_PRICE_DIR:-price}"
rm -rf "$OUT"
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    case "$driver_major" in
        ''|*[!0-9]*) ;;
        *) if [ "$driver_major" -lt 580 ] && [ -x /usr/local/cuda/bin/ptxas ]; then
               export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
           fi ;;
    esac
fi
red=0
others() {
    # What else is running on the box that could disturb a time: compilers
    # and other harness binaries. Recorded before and after the timed run.
    ps -eo pid,etime,comm,args 2>/dev/null | grep -E 'mojo|price|pixi' | grep -v -E "grep|$$|ps -eo" | cut -c1-200
}
{
    echo "profile=mojolearn.identical.gemm.int15i64.v1"
    echo "box=$BOX"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "tree_head=$(git rev-parse HEAD 2>/dev/null) dirty_files=$(git status --porcelain 2>/dev/null | grep -c .)"
    echo "mac_budget=${MOJOLEARN_INT15_PRICE_MAC_BUDGET:-default} repeats=${MOJOLEARN_INT15_PRICE_REPEATS:-default} only=${MOJOLEARN_INT15_PRICE_ONLY:-all}"
    if [ "$(uname -s)" = Darwin ]; then
        echo "machine=$(sysctl -n machdep.cpu.brand_string 2>/dev/null) macOS $(sw_vers -productVersion 2>/dev/null)"
    elif command -v nvidia-smi > /dev/null 2>&1; then
        echo "machine=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
    fi
} > "$OUT/run.txt"

t0=$(date +%s)
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/gemm_int15_price_main.mojo -o "$OUT/int15_price" > "$OUT/build.log" 2>&1
rc=$?
echo "build exit=$rc $(( $(date +%s) - t0 ))s" >> "$OUT/run.txt"
if [ "$rc" -ne 0 ]; then
    red=1
    cut -c1-400 "$OUT/build.log" | tail -40
else
    t0=$(date +%s)
    MOJOLEARN_INT15_PRICE_IDENTITY_ONLY=1 pixi run "$OUT/int15_price" > "$OUT/warm.log" 2>&1
    rc=$?
    echo "warm exit=$rc $(( $(date +%s) - t0 ))s (untimed, every arm once)" >> "$OUT/run.txt"
    [ "$rc" -eq 0 ] || red=1
    { echo "others_before_the_timed_run:"; others; } >> "$OUT/run.txt"
    t0=$(date +%s)
    pixi run "$OUT/int15_price" > "$OUT/price.log" 2>&1
    rc=$?
    echo "price exit=$rc $(( $(date +%s) - t0 ))s" >> "$OUT/run.txt"
    [ "$rc" -eq 0 ] || red=1
    { echo "others_after_the_timed_run:"; others; } >> "$OUT/run.txt"
fi
grep -E '^INT15(-NOT-RUN)? ' "$OUT/price.log" 2>/dev/null > "$OUT/int15.tsv"
grep -E '^DIGEST ' "$OUT/price.log" 2>/dev/null > "$OUT/digests.tsv"
grep -E '^DIGEST ' "$OUT/warm.log" 2>/dev/null > "$OUT/digests_warm.tsv"
if [ -s "$OUT/digests.tsv" ] && cmp -s "$OUT/digests.tsv" "$OUT/digests_warm.tsv"; then
    echo "warm-and-timed: the two runs printed the same digests ($(grep -c . "$OUT/digests.tsv"))" >> "$OUT/run.txt"
else
    echo "warm-and-timed: the two runs' digests DIFFER, or a run printed none" >> "$OUT/run.txt"
    red=1
fi

MOJOLEARN_INT15_PRICE_ONLY=${MOJOLEARN_INT15_PRICE_ONLY:-qkv} MOJOLEARN_INT15_PRICE_IDENTITY_ONLY=1 \
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_LOWBIT_SABOTAGE=1 \
    -I . bench/gemm_int15_price_main.mojo > "$OUT/price_sabotage.log" 2>&1
echo "price-sabotage exit=$? (identity only)" >> "$OUT/run.txt"
grep -E '^DIGEST ' "$OUT/price_sabotage.log" > "$OUT/digests_sabotage.tsv"
python3 tools/lowbit_int15/digests.py --expect-disagree clean="$OUT/digests.tsv" sabotage="$OUT/digests_sabotage.tsv" > "$OUT/sabotage_verdict.txt" 2>&1
rc=$?
echo "sabotage-comparison exit=$rc (0: every sabotaged fifteen-bit digest differed from the clean one)" >> "$OUT/run.txt"
[ "$rc" -eq 0 ] || red=1
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/run.txt"

cat "$OUT/run.txt"
echo "== price.log (every line that is not a PRICE, DIGEST or device-bits line)"
grep -v -E '^(PRICE|DIGEST|   device-bits) |mbind' "$OUT/price.log" 2>/dev/null | cut -c1-400
echo "== sabotage verdict"
cat "$OUT/sabotage_verdict.txt"
echo "== price digests (the timed run)"
cat "$OUT/digests.tsv"
echo "price_job: box=$BOX red=$red"
exit "$red"
