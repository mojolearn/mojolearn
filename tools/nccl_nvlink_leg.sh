#!/bin/bash
# WOUND DOWN 2026-09-12, NEVER RUN. DO NOT RUN THIS FOR "THE PRICE OF
# DETERMINISM" -- that framing was withdrawn by Andrew before this leg
# executed, and a better-measured number would not repair it.
#
# The reason is not the hardware, so a faster box does not fix it. Pricing
# determinism means comparing a deterministic arm against OUR OWN optimized
# NON-deterministic arm of the same operation. We have no such arm for
# all-reduce. Timing our fixed-order reduction against NCCL measures the SUM of
# three things -- the cost of fixing the summation order, the gap between our
# implementation quality and NVIDIA's, and the configuration each side happened
# to run -- and nothing in the experiment separates them. Any ratio it prints
# is therefore unfalsifiable as a statement about determinism, whether it comes
# out at 1.07x on PCIe or 3x on NVLink.
#
# What IS still worth measuring with this code is NCCL'S OWN BEHAVIOR, which is
# falsifiable and needs no arm of ours to compare against: whether a fixed
# configuration is bit-reproducible, whether Ring and Tree agree, and at which
# message sizes NCCL silently switches algorithm or protocol. Those live in
# --mode hashes and --mode oracle. --mode timing is the withdrawn part.
#
# Runs ON the rented 4-GPU NVLink node. Prices a deterministic all-reduce
# against NCCL on hardware where NCCL is NOT handicapped.
#
#   bash tools/nccl_nvlink_leg.sh <outdir>
#
# THE GATE COMES FIRST. The earlier lane's 1.07x number is confounded because
# CUDA peer-to-peer hung that PCIe host at >= 3 ranks, so every 4-rank case
# carried NCCL_P2P_DISABLE=1 and both arms fell through shared memory -- ring's
# topology advantage was erased and the ratio flattered determinism. This leg
# therefore REFUSES TO TIME ANYTHING until it has proven, on this box:
#   1. nvidia-smi topo -m shows NV# links between the GPUs, not just PHB/PIX;
#   2. every ordered GPU pair reports can_device_access_peer;
#   3. a 4-rank all-reduce with peer-to-peer ENABLED completes (no hang).
# If any of those fails the leg writes NVLINK_GATE=FAIL and exits non-zero.
# Producing a second confounded number is worse than producing none.
#
# Nothing below ever sets NCCL_P2P_DISABLE. That is the whole point.
set -u
OUT="${1:?outdir}"
mkdir -p "$OUT/logs"
cd "$(dirname "$0")/.." || exit 9
PROBE=tools/nccl_nvlink_probe.py
NG=$(nvidia-smi -L | wc -l | tr -d ' ')

# Sizes in BYTES PER RANK: 1 KB, 1 MB, 8 MB, 64 MB, 256 MB.
FULL="1024,1048576,8388608,67108864,268435456"
SMALL="1024,1048576,67108864"

{
  echo "== date ==";        date -u
  echo "== nvidia-smi ==";  nvidia-smi
  echo "== topology ==";    nvidia-smi topo -m
  echo "== nvlink status =="; nvidia-smi nvlink -s 2>&1 | head -60
  echo "== driver/cuda =="; nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv
  echo "== python ==";      python -c "import torch,sys;print('torch',torch.__version__);print('cuda',torch.version.cuda);print('nccl',torch.cuda.nccl.version());print('py',sys.version)"
  echo "== gpus ==";        echo "$NG"
} > "$OUT/stack.txt" 2>&1

# ---------------------------------------------------------------- the gate
GATE_OK=1
{
  echo "=== gate 1: NV links in nvidia-smi topo -m ==="
  nvidia-smi topo -m
  if nvidia-smi topo -m | grep -qE '(^|[[:space:]])NV[0-9]+([[:space:]]|$)'; then
      echo "GATE1=PASS (NV# links present)"
  else
      echo "GATE1=FAIL (no NV# links: this is a PCIe box, which is the confound we came to remove)"
  fi

  echo "=== gate 2: can_device_access_peer over every ordered pair ==="
  python - <<'PY'
import torch
n = torch.cuda.device_count()
bad = []
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        if not torch.cuda.can_device_access_peer(i, j):
            bad.append((i, j))
print("devices", n, "pairs without peer access:", bad)
print("GATE2=" + ("PASS" if not bad else "FAIL"))
PY

  echo "=== gate 3: 4-rank all-reduce with peer-to-peer ENABLED, 180 s cap ==="
  timeout 180 torchrun --standalone --nproc_per_node="$NG" "$PROBE" \
      --mode hashes --sizes-bytes 1048576 --dtypes float32 --iters 2 --tag gate_p2p_on
  rc=$?
  echo "gate3 rc=$rc"
  if [ "$rc" -eq 0 ]; then echo "GATE3=PASS"; else echo "GATE3=FAIL (rc=$rc; 124 means it hung)"; fi
} > "$OUT/logs/gate.log" 2>&1

grep -q 'GATE1=PASS' "$OUT/logs/gate.log" || GATE_OK=0
grep -q 'GATE2=PASS' "$OUT/logs/gate.log" || GATE_OK=0
grep -q 'GATE3=PASS' "$OUT/logs/gate.log" || GATE_OK=0
if [ "$GATE_OK" -ne 1 ]; then
    echo "NVLINK_GATE=FAIL" | tee -a "$OUT/stack.txt"
    echo "[leg] GATE FAILED -- refusing to time anything. See $OUT/logs/gate.log"
    exit 3
fi
echo "NVLINK_GATE=PASS" | tee -a "$OUT/stack.txt"
echo "[leg] gate passed: NV links present, peer access on every pair, 4-rank all-reduce completes with P2P ON"

run_case() {  # name world env... -- probe args...
  local name="$1" world="$2"; shift 2
  local envs=() ; while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local log="$OUT/logs/$name.log"
  echo "[leg] $(date -u +%T) $name (world=$world) ${envs[*]:-auto}"
  if [ ${#envs[@]} -eq 0 ]; then
      torchrun --standalone --nproc_per_node="$world" "$PROBE" --tag "$name" "$@" > "$log" 2>&1
  else
      env "${envs[@]}" torchrun --standalone --nproc_per_node="$world" "$PROBE" --tag "$name" "$@" > "$log" 2>&1
  fi
  local rc=$?
  echo "rc=$rc" >> "$log"
  python3 - "$log" "$name" "$world" "$rc" "$OUT/rows.jsonl" <<'PY'
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

# The pinned configuration of arm B. Rank order is pinned too: CUDA_VISIBLE_DEVICES
# fixes the device list and torchrun maps LOCAL_RANK to it identically every run.
PIN_FULL=(NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 CUDA_VISIBLE_DEVICES=0,1,2,3)
PIN_AP=(NCCL_ALGO=Ring NCCL_PROTO=Simple CUDA_VISIBLE_DEVICES=0,1,2,3)

# ---- 1. the witness: do these inputs expose non-associativity at 4 ranks?
run_case oracle_w$NG "$NG" -- --mode oracle --sizes-bytes "$SMALL"

# ---- 2. C2 is C1's arithmetic: same bits as C1, and the same bits at every
#         chunk count. If this fails, C2's timing rows mean nothing.
run_case invariance_w$NG "$NG" -- --mode invariance --sizes-bytes "$SMALL" --c2-chunks 1,4,8

# ---- 3. do the settled findings survive peer-to-peer being ENABLED?
#         (a) repeated runs and process restarts do not vary; (b) Ring != Tree.
for r in a b; do
  run_case p2pon_auto_w${NG}_$r "$NG" -- --mode hashes --sizes-bytes "$SMALL" --iters 15
  run_case p2pon_RingSimple_w${NG}_$r "$NG" NCCL_ALGO=Ring NCCL_PROTO=Simple -- --mode hashes --sizes-bytes "$SMALL" --iters 15
  run_case p2pon_TreeSimple_w${NG}_$r "$NG" NCCL_ALGO=Tree NCCL_PROTO=Simple -- --mode hashes --sizes-bytes "$SMALL" --iters 15
  run_case p2pon_RingLL128_w${NG}_$r "$NG" NCCL_ALGO=Ring NCCL_PROTO=LL128 -- --mode hashes --sizes-bytes "$SMALL" --iters 15
done

# ---- 4. what NCCL picked for itself, for the record (channels, algo, NVLS).
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,GRAPH \
  torchrun --standalone --nproc_per_node="$NG" "$PROBE" \
  --mode hashes --sizes-bytes "$FULL" --dtypes float32 --iters 2 --tag debug_info \
  > "$OUT/logs/nccl_debug_info.log" 2>&1
echo "rc=$?" >> "$OUT/logs/nccl_debug_info.log"

# ---- 5. THE PRICE. Two rounds of (A+C1+C2 interleaved, then B pinned), so the
#         cross-process A-vs-B comparison is measured twice and drift is visible.
#         Arms A, C1 and C2 are interleaved rep by rep inside one process.
#         Arm B can only be a separate process: NCCL reads its knobs at
#         communicator build time.
for round in 1 2; do
  run_case timing_auto_r${round}_w$NG "$NG" -- \
      --mode timing --sizes-bytes "$FULL" --arms nccl,c1,c2 --reps 15 --c2-chunks 4
  run_case timing_pinned_full_r${round}_w$NG "$NG" "${PIN_FULL[@]}" -- \
      --mode timing --sizes-bytes "$FULL" --arms nccl --reps 15
  run_case timing_pinned_algoproto_r${round}_w$NG "$NG" "${PIN_AP[@]}" -- \
      --mode timing --sizes-bytes "$FULL" --arms nccl --reps 15
done

echo "[leg] done $(date -u +%T); $(wc -l < "$OUT/rows.jsonl") rows in $OUT/rows.jsonl"
