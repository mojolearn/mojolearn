#!/bin/bash
# tools/lowbit_amd_tuned/scratch_first.sh -- lane/lowbit-amd-tuned: runs
# tools/lowbit_amd_tuned/scratch_first.mojo, with and without scratch, in
# `runs` fresh processes each (alternating), counts the device faults, and
# reads each binary's code object for the kernel's private segment, so the
# "scratch" binary is shown to use scratch and the other not to.
#
#   bash tools/lowbit_amd_tuned/scratch_first.sh [runs]
set -u
cd "$(dirname "$0")/../.." || exit 9
RUNS=${1:-150}
BOX=${MOJOLEARN_LOWBIT_BOX:-$(hostname -s)}
OUT="$PWD/bench/results/lowbit_amd_tuned/$BOX/scratch_first"
rm -rf "$OUT"
mkdir -p "$OUT/bin" "$OUT/co"
export PATH="$HOME/.pixi/bin:$PATH"
echo "box=$BOX started=$(date -u +%Y-%m-%dT%H:%M:%SZ) runs=$RUNS tree_head=$(git rev-parse HEAD 2>/dev/null)" > "$OUT/summary.txt"
pixi run mojo build -D MOJOLEARN_SCRATCH=1 -I . tools/lowbit_amd_tuned/scratch_first.mojo -o "$OUT/bin/scratch" > "$OUT/build_scratch.log" 2>&1
echo "build scratch exit=$?" >> "$OUT/summary.txt"
pixi run mojo build -I . tools/lowbit_amd_tuned/scratch_first.mojo -o "$OUT/bin/noscratch" > "$OUT/build_noscratch.log" 2>&1
echo "build noscratch exit=$?" >> "$OUT/summary.txt"
if [ ! -x "$OUT/bin/scratch" ] || [ ! -x "$OUT/bin/noscratch" ]; then
    echo "a build failed; nothing is run" >> "$OUT/summary.txt"
    tail -20 "$OUT"/build_*.log >> "$OUT/summary.txt"
    cat "$OUT/summary.txt"
    exit 1
fi
for kind in scratch noscratch; do
    python3 - "$OUT/bin/$kind" "$OUT/co/$kind" <<'PY'
import struct, sys
data = open(sys.argv[1], 'rb').read()
i = data.find(b'\x7fELF', 1); n = 0
while i > 0:
    try:
        if data[i + 4] == 2 and struct.unpack_from('<H', data, i + 18)[0] == 224:
            e_shoff = struct.unpack_from('<Q', data, i + 40)[0]
            e_shentsize, e_shnum = struct.unpack_from('<HH', data, i + 58)
            open('%s_%d.o' % (sys.argv[2], n), 'wb').write(data[i:i + e_shoff + e_shentsize * e_shnum]); n += 1
    except Exception:
        pass
    i = data.find(b'\x7fELF', i + 1)
PY
    for co in "$OUT"/co/${kind}_*.o; do
        [ -e "$co" ] || continue
        echo "$kind $(basename "$co"): $(/opt/rocm/llvm/bin/llvm-readelf --notes "$co" 2>/dev/null | grep -E 'private_segment_fixed_size|vgpr_count|vgpr_spill_count' | tr -s ' ' | tr '\n' ' ')" >> "$OUT/summary.txt"
    done
done
fs=0; fn=0; bad=0
for r in $(seq 1 "$RUNS"); do
    for kind in scratch noscratch; do
        "$OUT/bin/$kind" > "$OUT/last_$kind.log" 2>&1
        rc=$?
        if grep -q "Memory access fault" "$OUT/last_$kind.log" || [ "$rc" -ge 128 ]; then
            [ "$kind" = scratch ] && fs=$((fs + 1)) || fn=$((fn + 1))
            echo "run $r $kind FAULT exit=$rc: $(grep -m1 'Memory access fault' "$OUT/last_$kind.log")" >> "$OUT/summary.txt"
        elif [ "$rc" -ne 0 ]; then
            bad=$((bad + 1))
            echo "run $r $kind exit=$rc: $(tail -2 "$OUT/last_$kind.log" | tr '\n' ' ')" >> "$OUT/summary.txt"
        fi
    done
done
{
    echo "faults: scratch $fs of $RUNS, noscratch $fn of $RUNS; other failures $bad"
    echo "== kernel log fault lines (last 12)"
    (dmesg 2>/dev/null || journalctl -k --no-pager 2>/dev/null) | grep -i -E "retry page fault|for process|in page starting" | tail -12
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$OUT/summary.txt"
cat "$OUT/summary.txt"
rm -rf "$OUT/bin" "$OUT/co"
echo "scratch_first: box=$BOX scratch_faults=$fs noscratch_faults=$fn"
exit 0
