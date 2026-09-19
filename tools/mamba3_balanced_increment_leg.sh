#!/bin/sh
# Price the bit-identical 8x32 and 16x16 Mamba-3 state-increment tiles.
set -eu
out=${MOJOLEARN_GEMM_LEG_OUT:-/root/gemm_leg_out}
mkdir -p "$out"

pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
  mamba/checks/mamba3_increment_tile_check.mojo -o "$out/increment-check" \
  >"$out/build-check.log" 2>&1
"$out/increment-check" >"$out/increment-check.log" 2>&1

pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
  mamba/checks/mamba3_increment_tile_price.mojo -o "$out/increment-price" \
  >"$out/build-price.log" 2>&1
"$out/increment-price" >"$out/increment-price.log" 2>&1

python3 - "$out" <<'PY'
from pathlib import Path
import json, re, statistics, sys
out = Path(sys.argv[1])
summary = {}
text = (out / "increment-price.log").read_text()
for shape in ("narrow.b8_l4096_d512", "wide.b8_l1024_d2048"):
    summary[shape] = {}
    for arm in ("current_shared_v", "tiled_8x32", "balanced_16x16"):
        rows = []
        found = re.findall(rf"^M3_INCREMENT_PRICE {re.escape(shape)} round ([1-6]) arm {arm} ms ([0-9.]+)$", text, re.M)
        if len(found) != 6:
            raise SystemExit(f"missing increment timing: {shape} {arm}: {found}")
        rows = [float(ms) for _, ms in found]
        summary[shape][arm] = {"samples_ms": rows, "median_ms": statistics.median(rows)}
    summary[shape]["candidate_over_current"] = (
        summary[shape]["balanced_16x16"]["median_ms"] /
        summary[shape]["current_shared_v"]["median_ms"]
    )
    summary[shape]["candidate_over_tiled_8x32"] = (
        summary[shape]["balanced_16x16"]["median_ms"] /
        summary[shape]["tiled_8x32"]["median_ms"]
    )
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
