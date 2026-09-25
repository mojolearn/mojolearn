#!/bin/sh
# tools/nvidia_step_time/verify_lanes.sh -- lane/nvidia-step-time: the 201
# non-par lanes that reach gemm/checks/gemm_identical.mojo, in chunks of 25,
# on the installed byte LM binding and the device bindings of this commit,
# against the shipped reference table. Runs on the box; writes verify<tag>/.
set -u
cd /root/mojolearn || exit 9
PATH="$HOME/.pixi/bin:/usr/local/cuda/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=sm_90a
export PYTHONPATH=/root/mojolearn/python:/root/mojolearn
OUT=/root/gemm_leg_out/nv-step-time/verify${1:-}
mkdir -p "$OUT"
LANES=tools/nvidia_step_time/lanes_gemm_nonpar.txt
[ -s "$LANES" ] || { echo "no lane list at $LANES"; exit 3; }
rm -f /tmp/nvlanechunk.*
split -l 25 "$LANES" /tmp/nvlanechunk.
for f in /tmp/nvlanechunk.*; do
    c=${f##*.}
    t0=$(date +%s)
    pixi run python -m mojolearn verify --lanes "$(tr '\n' ',' < "$f" | sed 's/,$//')" --json-out "$OUT/chunk-$c.json" > "$OUT/chunk-$c.log" 2>&1
    echo "$(date -u +%H:%M:%S) verify chunk $c exit=$? secs=$(( $(date +%s) - t0 )): $(grep '^RESULT' "$OUT/chunk-$c.log" | tail -1 | cut -c1-260)" | tee -a "$OUT/summary.txt"
    gzip -9 -f "$OUT/chunk-$c.json"
done
echo "$(date -u +%H:%M:%S) verify done" | tee -a "$OUT/summary.txt"
