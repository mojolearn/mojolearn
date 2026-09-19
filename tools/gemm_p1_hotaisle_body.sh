#!/bin/sh
# MI300X proof and price for the production P == 1 tuned-GEMM specialization.
set -eu

ROOT=${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_GEMM_LEG_OUT:-/root/gemm_leg_out/gemm-p1}
JOBS=${MOJOLEARN_COMPILE_JOBS:-2}
SHAPES=pca.transform.wide.8192x64x128,kmeans.dist.4096x64x64
mkdir -p "$OUT/bin"
cd "$ROOT"

pixi install > "$OUT/pixi-install.log" 2>&1
pixi run mojo --version > "$OUT/mojo-version.txt" 2>&1
rocminfo > "$OUT/rocminfo.txt" 2>&1 || true
rocm-smi --showproductname --showmeminfo vram > "$OUT/rocm-smi-before.txt" 2>&1 || true

COMMON="-j $JOBS --target-accelerator gfx942 -D MOJOLEARN_COLUMN_AMD -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I ."
# shellcheck disable=SC2086
pixi run mojo build $COMMON gemm/checks/gemm_device_check.mojo \
    -o "$OUT/bin/device-check" > "$OUT/build-device-check.log" 2>&1
# shellcheck disable=SC2086
pixi run mojo build $COMMON gemm/checks/gemm_tuned_probe.mojo \
    -o "$OUT/bin/candidate" > "$OUT/build-candidate.log" 2>&1
# Same source and toolchain; this define restores the former FS=16 dispatch.
# shellcheck disable=SC2086
pixi run mojo build $COMMON -D MOJOLEARN_GEMM_P1_FOLD_STACK_CONTROL=1 \
    gemm/checks/gemm_tuned_probe.mojo -o "$OUT/bin/control" \
    > "$OUT/build-control.log" 2>&1

timeout 900 "$OUT/bin/device-check" > "$OUT/device-check.log" 2>&1

: > "$OUT/order.txt"
r=0
while [ "$r" -lt 7 ]; do
    if [ $((r % 2)) -eq 0 ]; then order="control candidate"; else order="candidate control"; fi
    echo "round=$r order=$order" >> "$OUT/order.txt"
    for arm in $order; do
        env MOJOLEARN_SPEED_SHAPES="$SHAPES" MOJOLEARN_SPEED_ROUNDS=5 \
            MOJOLEARN_GEMM_BASELINE_PLAN=-2 \
            timeout 300 "$OUT/bin/$arm" > "$OUT/$arm.$r.log" 2>&1
    done
    r=$((r + 1))
done

python3 - "$OUT" <<'PY' > "$OUT/summary.txt"
import re, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
shapes = (
    "pca.transform.wide.8192x64x128",
    "kmeans.dist.4096x64x64",
)
rows = {arm: {shape: [] for shape in shapes} for arm in ("control", "candidate")}
digests = {arm: {shape: set() for shape in shapes} for arm in rows}
for arm in rows:
    for log in sorted(root.glob(f"{arm}.*.log")):
        text = log.read_text()
        for shape in shapes:
            dm = re.search(rf"^DIGEST {re.escape(shape)} baseline=(0x[0-9a-f]+) candidate=(0x[0-9a-f]+)", text, re.M)
            if not dm or dm.group(1) != dm.group(2):
                raise SystemExit(f"missing or internally moved digest: {log} {shape}")
            digests[arm][shape].add(dm.group(1))
            samples = re.findall(rf"^SAMPLE {re.escape(shape)} .* candidate_ns=([0-9]+)$", text, re.M)
            if len(samples) != 5:
                raise SystemExit(f"wrong sample count: {log} {shape}: {len(samples)}")
            rows[arm][shape].extend(map(int, samples))
for shape in shapes:
    if digests["control"][shape] != digests["candidate"][shape] or len(digests["candidate"][shape]) != 1:
        raise SystemExit(f"cross-build digest mismatch: {shape} {digests}")
    c = statistics.median(rows["control"][shape]) / 1e6
    n = statistics.median(rows["candidate"][shape]) / 1e6
    print(f"P1_RESULT shape={shape} digest={next(iter(digests['candidate'][shape]))} "
          f"samples=35 control_ms={c:.6f} candidate_ms={n:.6f} ratio={n/c:.6f}")
print("P1_BITS cross_build=IDENTICAL internal=IDENTICAL")
PY

rocm-smi --showproductname --showmeminfo vram > "$OUT/rocm-smi-after.txt" 2>&1 || true
cat "$OUT/summary.txt"
