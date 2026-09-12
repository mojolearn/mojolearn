#!/bin/bash
# Runs ON the rented multi-GPU node. Drives tools/nccl_determinism_probe.py
# over the configurations that are supposed to change NCCL's reduction order,
# and writes one log per configuration plus a JSONL of every hash.
#
#   bash tools/nccl_determinism_leg.sh <outdir>
#
# Every configuration is a SEPARATE torchrun process, because NCCL reads
# NCCL_ALGO / NCCL_PROTO / channel counts when the communicator is built. That
# also makes the process-restart test free: each config is run twice, as two
# independent processes with identical inputs.
set -u
OUT="${1:?outdir}"
mkdir -p "$OUT/logs"
cd "$(dirname "$0")/.." || exit 9
PROBE=tools/nccl_determinism_probe.py
NG=$(nvidia-smi -L | wc -l | tr -d ' ')

{
  echo "== date ==";           date -u
  echo "== nvidia-smi ==";     nvidia-smi
  echo "== topology ==";       nvidia-smi topo -m
  echo "== driver/cuda ==";    nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv
  echo "== python ==";         python -c "import torch,sys;print('torch',torch.__version__);print('cuda',torch.version.cuda);print('nccl',torch.cuda.nccl.version());print('py',sys.version)"
  echo "== nccl lib ==";       python -c "import torch,os;print(os.path.realpath(torch.__file__))"; ls -l /usr/lib/x86_64-linux-gnu/libnccl* 2>/dev/null
  echo "== gpus ==";           echo "$NG"
} > "$OUT/stack.txt" 2>&1

run_case() {  # name world env... -- probe args...
  local name="$1" world="$2"; shift 2
  local envs=() ; while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local log="$OUT/logs/$name.log"
  echo "[leg] $name (world=$world) ${envs[*]}"
  env "${envs[@]}" torchrun --standalone --nproc_per_node="$world" "$PROBE" --tag "$name" "$@" \
      > "$log" 2>&1
  local rc=$?
  echo "rc=$rc" >> "$log"
  python3 - "$log" "$name" "$world" "$rc" "$OUT/hashes.jsonl" <<'PY'
import json, sys
log, name, world, rc, out = sys.argv[1:]
rows = 0
with open(out, "a") as fh:
    for line in open(log, errors="replace"):
        if line.startswith("JSON|"):
            d = json.loads(line[5:])
            d["case"] = name; d["world_launched"] = int(world); d["rc"] = int(rc)
            fh.write(json.dumps(d, sort_keys=True) + "\n"); rows += 1
    if rows == 0:
        fh.write(json.dumps({"case": name, "world_launched": int(world),
                             "rc": int(rc), "failed": True}) + "\n")
print(f"[leg]   {rows} json rows, rc={rc}")
PY
}

SMALL="256,262144,16777216"            # 1 KB, 1 MB, 64 MB per rank
FULL="256,16384,262144,2097152,16777216,67108864"   # 1 KB .. 256 MB

# 1. The witness: do these inputs actually expose non-associativity?
run_case oracle_w$NG "$NG" -- -- --mode oracle --sizes "$SMALL"

# 2. Size sweep at NCCL's own choices, twice (process-restart test).
for r in a b; do
  run_case auto_full_w${NG}_$r "$NG" -- -- --mode hashes --sizes "$FULL" --iters 25
done

# 3. Forced algorithm and protocol, twice each.
for algo in Ring Tree; do
  for proto in Simple LL LL128; do
    for r in a b; do
      run_case ${algo}_${proto}_w${NG}_$r "$NG" \
        NCCL_ALGO=$algo NCCL_PROTO=$proto -- --mode hashes --sizes "$SMALL" --iters 15
    done
  done
done

# 4. Channel count: a tuning knob NCCL derives from the topology.
for nch in 1 2; do
  for r in a b; do
    run_case chan${nch}_w${NG}_$r "$NG" \
      NCCL_MIN_NCHANNELS=$nch NCCL_MAX_NCHANNELS=$nch -- --mode hashes --sizes "$SMALL" --iters 15
  done
done

# 5. Device count: the same buffers reduced over fewer ranks.
if [ "$NG" -gt 2 ]; then
  for r in a b; do
    run_case auto_w2_$r 2 -- -- --mode hashes --sizes "$SMALL" --iters 15
  done
  run_case oracle_w2 2 -- -- --mode oracle --sizes "$SMALL"
fi

# 6. What NCCL actually picked, for the record.
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING torchrun --standalone --nproc_per_node="$NG" \
  "$PROBE" --mode hashes --sizes "$FULL" --iters 2 --tag debug_info \
  > "$OUT/logs/nccl_debug_info.log" 2>&1
echo "rc=$?" >> "$OUT/logs/nccl_debug_info.log"

# 7. The price of determinism. Timed arms run one at a time, nothing else on
#    the GPUs, NCCL and the fixed-order all-reduce never overlapping.
run_case timing_w$NG "$NG" -- -- --mode timing --sizes "$FULL" --reps 15

echo "[leg] done; $(wc -l < "$OUT/hashes.jsonl") hash rows in $OUT/hashes.jsonl"
