#!/bin/bash
# tools/lowbit_amd_tuned/fault_stress.sh -- lane/lowbit-amd-tuned: the GPU
# memory access fault, second step. Job 1790659053624 (fault_repro.sh) saw it
# once in 24 runs, in a build with the loads' alignment STATED (so not the
# byte path), on the FIRST launch in the process of the direct plan of 64x64
# per wave (`direct.w64x64.b128x128.l16.u1`) at the first shape (1x32x32); the
# same binary passed its two other runs, and every build of one source had
# one sha256. So it is a property of a RUN, not of a compile.
#
#   bash tools/lowbit_amd_tuned/fault_stress.sh [runs]
#
# 1. Builds the gate once (stated loads, every launch named) and runs it
#    `runs` times: how often, and at which launch.
# 2. Reads every AMD code object embedded in that binary (ELF, EM_AMDGPU)
#    and prints each kernel's registers, spills and private (scratch)
#    segment: whether the kernel that faults is the first to need scratch.
# Launches only the gate; times nothing. On AMD only.
set -u
cd "$(dirname "$0")/../.." || exit 9
RUNS=${1:-25}
BOX=${MOJOLEARN_LOWBIT_BOX:-$(hostname -s)}
OUT="$PWD/bench/results/lowbit_amd_tuned/$BOX/fault_stress"
rm -rf "$OUT"
mkdir -p "$OUT/bin" "$OUT/co"
export PATH="$HOME/.pixi/bin:$PATH"
if ! { command -v rocm-smi > /dev/null 2>&1 || command -v amd-smi > /dev/null 2>&1; }; then
    echo "fault_stress: box=$BOX is not an AMD box. NOT RUN." | tee "$OUT/summary.txt"
    exit 3
fi
bin="$OUT/bin/gate"
{
    echo "box=$BOX started=$(date -u +%Y-%m-%dT%H:%M:%SZ) runs=$RUNS tree_head=$(git rev-parse HEAD 2>/dev/null)"
    echo "HSA_XNACK=${HSA_XNACK:-unset} $(cat /sys/module/amdgpu/parameters/noretry 2>/dev/null | sed 's/^/amdgpu.noretry=/')"
} > "$OUT/summary.txt"
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_INT8_AMD_TRACE=1 -I . \
    gemm/checks/gemm_int8_mma_amd_check.mojo -o "$bin" > "$OUT/build.log" 2>&1
echo "build exit=$? sha256=$(sha256sum "$bin" 2>/dev/null | cut -c1-16)" >> "$OUT/summary.txt"
faults=0
for r in $(seq 1 "$RUNS"); do
    log="$OUT/run_$r.log"
    t0=$(date +%s)
    "$bin" > "$log" 2>&1
    rc=$?
    line="run $r exit=$rc $(( $(date +%s) - t0 ))s $(grep -E '^== [0-9]+ gates' "$log" | tail -1)"
    if grep -q "Memory access fault" "$log" || [ "$rc" -ge 128 ]; then
        faults=$((faults + 1))
        line="$line FAULT after: $(grep '^   launch ' "$log" | tail -1 | sed 's/^   launch //') launches-before=$(grep -c '^   launch ' "$log") | $(grep 'Memory access fault' "$log" | head -1)"
    fi
    echo "$line" >> "$OUT/summary.txt"
done
echo "faults=$faults of $RUNS" >> "$OUT/summary.txt"
# 2. The code objects.
READELF=""
for c in /opt/rocm/llvm/bin/llvm-readelf "$(ls -d /opt/rocm-*/llvm/bin/llvm-readelf 2>/dev/null | head -1)" "$PWD/.pixi/envs/default/bin/llvm-readelf" "$(command -v llvm-readelf)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then READELF=$c; break; fi
done
python3 - "$bin" "$OUT/co" <<'PY' >> "$OUT/summary.txt"
import struct, sys
data = open(sys.argv[1], 'rb').read()
out = sys.argv[2]
i = data.find(b'\x7fELF', 1)
n = 0
while i > 0:
    try:
        if data[i + 4] == 2:
            e_machine = struct.unpack_from('<H', data, i + 18)[0]
            e_shoff = struct.unpack_from('<Q', data, i + 40)[0]
            e_shentsize, e_shnum = struct.unpack_from('<HH', data, i + 58)
            end = e_shoff + e_shentsize * e_shnum
            if e_machine == 224 and 0 < end < 1 << 30:
                open('%s/co_%03d.o' % (out, n), 'wb').write(data[i:i + end])
                n += 1
    except Exception:
        pass
    i = data.find(b'\x7fELF', i + 1)
print('code objects (EM_AMDGPU) found in the binary: %d' % n)
PY
if [ -n "$READELF" ]; then
    echo "readelf=$READELF" >> "$OUT/summary.txt"
    for co in "$OUT"/co/*.o; do
        [ -e "$co" ] || continue
        "$READELF" --notes "$co" 2>/dev/null | grep -E '\.name:|\.symbol:|private_segment_fixed_size|vgpr_count|agpr_count|sgpr_count|spill_count|uses_dynamic_stack|group_segment_fixed_size' \
            | grep -v 'args' >> "$OUT/kernels.txt"
    done
    python3 - "$OUT/kernels.txt" <<'PY' >> "$OUT/summary.txt"
import re, sys
cur = {}
rows = []
for line in open(sys.argv[1], errors='replace'):
    m = re.match(r'\s*\.(\w+):\s*(.*)', line)
    if not m:
        continue
    k, v = m.group(1), m.group(2).strip()
    if k == 'name' and not v.startswith('.'):
        if cur:
            rows.append(cur)
        cur = {'name': v}
    elif cur:
        cur[k] = v
if cur:
    rows.append(cur)
print('== kernels: name | vgpr | agpr | sgpr | vgpr spill | sgpr spill | private (scratch) bytes | dynamic stack | group bytes')
for r in rows:
    nm = r['name']
    print(' | '.join([nm[:150], r.get('vgpr_count', '?'), r.get('agpr_count', '?'), r.get('sgpr_count', '?'),
                      r.get('vgpr_spill_count', '?'), r.get('sgpr_spill_count', '?'),
                      r.get('private_segment_fixed_size', '?'), r.get('uses_dynamic_stack', '?'),
                      r.get('group_segment_fixed_size', '?')]))
PY
else
    echo "no llvm-readelf on the box: the kernels' scratch is NOT read" >> "$OUT/summary.txt"
fi
{
    echo "== kernel log lines about the GPU faults since the job began (if readable)"
    (dmesg 2>/dev/null || journalctl -k --no-pager 2>/dev/null) | grep -i -E "page fault|for process|in page starting|fault from" | tail -24
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$OUT/summary.txt"
cat "$OUT/summary.txt"
rm -rf "$OUT/bin" "$OUT/co"
echo "fault_stress: box=$BOX faults=$faults"
exit 0
