#!/bin/bash
# tools/lowbit_units/price_job.sh -- lane/lowbit-units, the timing harness
# (bench/gemm_lowbit_price_main.mojo) on one box, then its sabotage arm.
#
#   price            the clean IDENTICAL build, every OP_NT transformer row.
#                    EXPECTED exit 0.
#   price-sabotage   -D MOJOLEARN_LOWBIT_PRICE_SABOTAGE=1 (one bit of one
#                    cell of every arm's output flipped before the digest),
#                    one timed call per arm, the qkv rows only. Its digests
#                    must DIFFER from the clean run's at every arm, which
#                    tools/lowbit_units/table.py --expect-disagree checks
#                    here. EXPECTED: the comparison exits 0 (every digest
#                    differed).
#
# Writes bench/results/lowbit_units/<box>/price/ and prints the LOWBIT lines
# and the verdicts, so a steward's stdout carries them home. The mac budget
# and the repeats come from the environment (bench/gemm_lowbit_price_main.mojo);
# THE SAME BUDGET MUST RUN ON EVERY BOX whose hashes are compared.
set -u
cd "$(dirname "$0")/../.." || exit 9
BOX=${MOJOLEARN_LOWBIT_BOX:-$(hostname -s)}
OUT="$PWD/bench/results/lowbit_units/$BOX/price"
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    # tools/lowbit_mma_leg.sh's lines: an older driver uses its own assembler.
    driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    case "$driver_major" in
        ''|*[!0-9]*) ;;
        *) if [ "$driver_major" -lt 580 ] && [ -x /usr/local/cuda/bin/ptxas ]; then
               export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
           fi ;;
    esac
fi
red=0
{
    echo "box=$BOX"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit=$(git rev-parse HEAD 2>/dev/null)"
    echo "mac_budget=${MOJOLEARN_LOWBIT_PRICE_MAC_BUDGET:-default} repeats=${MOJOLEARN_LOWBIT_PRICE_REPEATS:-default} only=${MOJOLEARN_LOWBIT_PRICE_ONLY:-all}"
} > "$OUT/run.txt"

pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/gemm_lowbit_price_main.mojo > "$OUT/price.log" 2>&1
rc=$?
echo "price exit=$rc" >> "$OUT/run.txt"
[ "$rc" -eq 0 ] || red=1
grep -E '^LOWBIT(-NOT-RUN)? ' "$OUT/price.log" > "$OUT/lowbit.tsv"

MOJOLEARN_LOWBIT_PRICE_ONLY=qkv MOJOLEARN_LOWBIT_PRICE_REPEATS=1 \
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_LOWBIT_PRICE_SABOTAGE=1 \
    -I . bench/gemm_lowbit_price_main.mojo > "$OUT/price_sabotage.log" 2>&1
echo "price-sabotage exit=$? (the in-run plan comparison may raise; the verdict is the digest comparison below)" >> "$OUT/run.txt"
grep -E '^LOWBIT ' "$OUT/price_sabotage.log" > "$OUT/lowbit_sabotage.tsv"
python3 tools/lowbit_units/table.py --expect-disagree "$OUT/lowbit.tsv" "$OUT/lowbit_sabotage.tsv" > "$OUT/sabotage_verdict.txt" 2>&1
rc=$?
echo "sabotage-comparison exit=$rc (0: every sabotaged digest differed from the clean one)" >> "$OUT/run.txt"
[ "$rc" -eq 0 ] || red=1
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/run.txt"

cat "$OUT/run.txt"
echo "== price.log (every line that is not a PRICE or device-bits line)"
grep -v -E '^(PRICE|   device-bits) ' "$OUT/price.log"
echo "== sabotage verdict"
cat "$OUT/sabotage_verdict.txt"
echo "price_job: box=$BOX red=$red"
exit "$red"
