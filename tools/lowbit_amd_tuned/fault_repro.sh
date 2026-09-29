#!/bin/bash
# tools/lowbit_amd_tuned/fault_repro.sh -- lane/lowbit-amd-tuned: does the
# GPU memory access fault of MI325X job 1790657510941 (the AMD gate built
# with -D MOJOLEARN_INT8_TUNED_UNSTATED=1, exit 139, "Memory access fault by
# GPU node-1", not seen again in jobs 1790657862351 and 1790658495381) come
# back, and is it a property of ONE BUILD (the gfx942 code the compiler
# emitted that time) or of ONE RUN (where the buffers landed)?
#
#   bash tools/lowbit_amd_tuned/fault_repro.sh [builds] [runs]
#
# Builds the gate `builds` times with the byte path forced, each with a
# distinct unused define (-D MOJOLEARN_REPRO_SALT=<i>) so no build replays
# another's cached compile, and the same number with the stated loads; runs
# every binary `runs` times. Every launch is named, flushed, before it is
# enqueued (-D MOJOLEARN_INT8_AMD_TRACE=1), so a fault names its launch.
# Also: the kernel driver's log of a fault, when the box lets it be read.
# Launches only the gate; times nothing. On AMD only.
set -u
cd "$(dirname "$0")/../.." || exit 9
BUILDS=${1:-4}
RUNS=${2:-3}
BOX=${MOJOLEARN_LOWBIT_BOX:-$(hostname -s)}
OUT="$PWD/bench/results/lowbit_amd_tuned/$BOX/fault_repro"
rm -rf "$OUT"
mkdir -p "$OUT/bin"
export PATH="$HOME/.pixi/bin:$PATH"
if ! { command -v rocm-smi > /dev/null 2>&1 || command -v amd-smi > /dev/null 2>&1; }; then
    echo "fault_repro: box=$BOX is not an AMD box. NOT RUN." | tee "$OUT/summary.txt"
    exit 3
fi
{
    echo "box=$BOX started=$(date -u +%Y-%m-%dT%H:%M:%SZ) builds=$BUILDS runs=$RUNS"
    echo "tree_head=$(git rev-parse HEAD 2>/dev/null)"
} > "$OUT/summary.txt"
faults=0
for kind in unstated stated; do
    for b in $(seq 1 "$BUILDS"); do
        extra=""
        [ "$kind" = unstated ] && extra="-D MOJOLEARN_INT8_TUNED_UNSTATED=1"
        bin="$OUT/bin/${kind}_$b"
        t0=$(date +%s)
        # shellcheck disable=SC2086
        pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 $extra -D MOJOLEARN_INT8_AMD_TRACE=1 \
            -D MOJOLEARN_REPRO_SALT=$b -I . gemm/checks/gemm_int8_mma_amd_check.mojo -o "$bin" \
            > "$OUT/build_${kind}_$b.log" 2>&1
        brc=$?
        echo "build $kind $b exit=$brc $(( $(date +%s) - t0 ))s sha256=$(sha256sum "$bin" 2>/dev/null | cut -c1-16)" >> "$OUT/summary.txt"
        [ "$brc" -eq 0 ] || continue
        for r in $(seq 1 "$RUNS"); do
            log="$OUT/run_${kind}_${b}_$r.log"
            t0=$(date +%s)
            "$bin" > "$log" 2>&1
            rc=$?
            line="run $kind build=$b run=$r exit=$rc $(( $(date +%s) - t0 ))s $(grep -E '^== [0-9]+ gates' "$log" | tail -1)"
            if [ "$rc" -ne 0 ] && ! grep -q -E '^== [0-9]+ gates, 0 failed ==$' "$log"; then
                if grep -q "Memory access fault" "$log" || [ "$rc" -ge 128 ]; then
                    faults=$((faults + 1))
                    line="$line FAULT after: $(grep '^   launch ' "$log" | tail -1)"
                    grep -i -E "fault|page|vmid|address" "$log" | head -5 >> "$OUT/summary.txt"
                fi
            fi
            echo "$line" >> "$OUT/summary.txt"
        done
    done
done
{
    echo "faults=$faults"
    echo "== kernel log lines about the GPU (if readable)"
    (dmesg 2>/dev/null || journalctl -k --no-pager 2>/dev/null) | grep -i -E "amdgpu|gfxhub|vm_fault|page fault" | tail -20
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$OUT/summary.txt"
cat "$OUT/summary.txt"
rm -rf "$OUT/bin"
echo "fault_repro: box=$BOX faults=$faults"
exit 0
