#!/bin/sh
# NVIDIA/AMD qualification body for the exact large RMSNorm backward fusion.
set -eu

OUT=${MOJOLEARN_GEMM_LEG_OUT_REMOTE:-/root/gemm_leg_out}
mkdir -p "$OUT/rmsnorm_cv"
CV="$OUT/rmsnorm_cv"

pixi run mojo build -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
  -D MOJOLEARN_BWD_NORM_SPLIT_TRIAL=1 -I . \
  bench/samba_rms_price_main.mojo -o /tmp/rmsnorm-split
pixi run mojo build -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
  -D MOJOLEARN_BWD_NORM_FUSED_TRIAL=1 -I . \
  bench/samba_rms_price_main.mojo -o /tmp/rmsnorm-fused

# Alternate whole processes. Each process contains seven synchronized calls
# per shape; process alternation keeps thermal/drift order from naming a winner.
i=1
while [ "$i" -le 5 ]; do
  /tmp/rmsnorm-split > "$CV/split-$i.log"
  /tmp/rmsnorm-fused > "$CV/fused-$i.log"
  i=$((i + 1))
done

pixi run mojo build -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
  -D MOJOLEARN_BWD_NORM_SPLIT_TRIAL=1 -I . \
  transformer/checks/transformer_backward_check.mojo -o /tmp/rmsnorm-split-check
pixi run mojo build -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
  -D MOJOLEARN_BWD_NORM_FUSED_TRIAL=1 -I . \
  transformer/checks/transformer_backward_check.mojo -o /tmp/rmsnorm-fused-check
/tmp/rmsnorm-split-check > "$CV/split-check.log"
sha256sum /tmp/mojolearn_transformer_backward.trace > "$CV/split-trace.sha256"
/tmp/rmsnorm-fused-check > "$CV/fused-check.log"
sha256sum /tmp/mojolearn_transformer_backward.trace > "$CV/fused-trace.sha256"

python3 - "$CV" <<'PY'
import json, pathlib, statistics, sys
p = pathlib.Path(sys.argv[1])
out = {}
for shape in ("512x64", "2048x512", "8192x768", "32768x768"):
    row = {}
    for arm in ("split", "fused"):
        process_medians = []
        for log in sorted(p.glob(f"{arm}-*.log")):
            values = []
            for line in log.read_text().splitlines():
                a = line.split()
                if len(a) == 6 and a[0] == "PRICE" and a[3] == shape and a[4] == "kernel.backward":
                    values.append(float(a[5]))
            if values:
                process_medians.append(statistics.median(values))
        row[arm] = {"process_medians_ms": process_medians,
                    "median_ms": statistics.median(process_medians)}
    row["speedup"] = row["split"]["median_ms"] / row["fused"]["median_ms"]
    out[shape] = row
(p / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY
