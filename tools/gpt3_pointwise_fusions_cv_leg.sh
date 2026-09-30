#!/bin/sh
# Qualify exact gated-SiLU and norm2/residual backward fusions on one GPU.
set -eu

OUT=${MOJOLEARN_GEMM_LEG_OUT_REMOTE:-/root/gemm_leg_out}
CV="$OUT/gpt3_pointwise_fusions_cv"
mkdir -p "$CV"

pixi run mojo build -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
  bench/gated_silu_backward_price_main.mojo -o /tmp/gated-silu-price
pixi run mojo build -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
  bench/norm2_residual_backward_price_main.mojo -o /tmp/norm2-residual-price

i=1
while [ "$i" -le 5 ]; do
  /tmp/gated-silu-price > "$CV/gated-$i.log"
  MOJOLEARN_NORM2_RESIDUAL_ARM=split /tmp/norm2-residual-price > "$CV/norm-split-$i.log"
  MOJOLEARN_NORM2_RESIDUAL_ARM=fused /tmp/norm2-residual-price > "$CV/norm-fused-$i.log"
  i=$((i + 1))
done

check_arm() {
  name=$1
  define=$2
  pixi run mojo build -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
    -D "$define=1" -I . transformer/checks/transformer_backward_check.mojo \
    -o "/tmp/$name-check"
  "/tmp/$name-check" > "$CV/$name-check.log"
  sha256sum /tmp/mojolearn_transformer_backward.trace > "$CV/$name-trace.sha256"
}
check_arm gated-split MOJOLEARN_BWD_GATED_SILU_SPLIT_TRIAL
check_arm gated-fused MOJOLEARN_BWD_GATED_SILU_FUSED_TRIAL
check_arm norm-split MOJOLEARN_BWD_NORM2_RESIDUAL_SPLIT_TRIAL
check_arm norm-fused MOJOLEARN_BWD_NORM2_RESIDUAL_FUSED_TRIAL

python3 - "$CV" <<'PY'
import json, pathlib, statistics, sys
p = pathlib.Path(sys.argv[1])
out = {"gated_silu": {}, "norm2_residual": {}}
for shape in ("2048x3072", "8192x3072", "32768x3072"):
    split, fused = [], []
    for log in sorted(p.glob("gated-*.log")):
        for line in log.read_text().splitlines():
            a = line.split()
            if len(a) == 6 and a[0] == "PRICE" and a[1] == shape:
                split.append(float(a[3])); fused.append(float(a[5]))
    out["gated_silu"][shape] = {
        "split_median_ms": statistics.median(split),
        "fused_median_ms": statistics.median(fused),
        "speedup": statistics.median(split) / statistics.median(fused),
        "samples": len(split),
    }
for shape in ("2048x768", "8192x768", "32768x768"):
    row = {}
    for arm in ("split", "fused"):
        process_medians = []
        for log in sorted(p.glob(f"norm-{arm}-[0-9].log")):
            vals = []
            for line in log.read_text().splitlines():
                a = line.split()
                if len(a) >= 4 and a[0] == "PRICE" and a[2] == shape:
                    vals.append(float(a[3]))
            process_medians.append(statistics.median(vals))
        row[arm + "_median_ms"] = statistics.median(process_medians)
        row[arm + "_process_medians_ms"] = process_medians
    row["speedup"] = row["split_median_ms"] / row["fused_median_ms"]
    out["norm2_residual"][shape] = row
(p / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY
