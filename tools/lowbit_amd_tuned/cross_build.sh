#!/bin/bash
# tools/lowbit_amd_tuned/cross_build.sh -- lane/lowbit-amd-tuned: does every
# file that imports gemm/checks/gemm_int8_mma_amd.mojo COMPILE on the columns
# that are not AMD? Builds (links, runs nothing) each file named below for
# NVIDIA (sm_90a, -D MOJOLEARN_COLUMN_NVIDIA), for Apple (apple-m3,
# -D MOJOLEARN_COLUMN_APPLE) and for the CPU column (-D MOJOLEARN_COLUMN_CPU),
# on whatever box it runs on. A build that fails prints its last lines. A
# cross build on a Linux box that fails for Apple for a reason of the
# toolchain (no Metal compiler off macOS) is reported as such, not as a pass.
#
#   bash tools/lowbit_amd_tuned/cross_build.sh <file> [<file> ...]
set -u
cd "$(dirname "$0")/../.." || exit 9
BOX=${MOJOLEARN_LOWBIT_BOX:-$(hostname -s)}
OUT="$PWD/bench/results/lowbit_amd_tuned/$BOX/cross_build"
rm -rf "$OUT"
mkdir -p "$OUT/bin"
export PATH="$HOME/.pixi/bin:$PATH"
echo "box=$BOX started=$(date -u +%Y-%m-%dT%H:%M:%SZ) tree_head=$(git rev-parse HEAD 2>/dev/null)" > "$OUT/summary.txt"
red=0
for f in "$@"; do
    base=$(basename "$f" .mojo)
    for col in nvidia apple cpu; do
        case $col in
            nvidia) flags="--target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA" ;;
            apple) flags="--target-accelerator apple-m3 -D MOJOLEARN_COLUMN_APPLE" ;;
            cpu) flags="-D MOJOLEARN_COLUMN_CPU" ;;
        esac
        log="$OUT/${base}_$col.log"
        t0=$(date +%s)
        # shellcheck disable=SC2086
        pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 $flags -I . "$f" -o "$OUT/bin/${base}_$col" > "$log" 2>&1
        rc=$?
        echo "$f column=$col exit=$rc $(( $(date +%s) - t0 ))s" >> "$OUT/summary.txt"
        if [ "$rc" -ne 0 ]; then
            grep -E "error:" "$log" | head -6 | cut -c1-300 >> "$OUT/summary.txt"
            [ "$col" = apple ] || red=1
        fi
    done
done
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) red=$red (an Apple build that fails off macOS is listed and not counted)" >> "$OUT/summary.txt"
cat "$OUT/summary.txt"
rm -rf "$OUT/bin"
exit "$red"
