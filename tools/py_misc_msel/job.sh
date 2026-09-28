#!/bin/bash
# lane/py-misc-msel: ONE light job on a shared NVIDIA pod (tools/nvidia_central.sh submit).
#  1. algos_lane_check x-metrics-splitters,x-metrics-search: builds what is stale, GPU == CPU
#  2. the same two lanes before (MOJOLEARN_MSEL_BEFORE=1) and after, GPU and CPU: hashes equal
#  3. check.py equal on GPU and CPU; 4. check.py time (1M rows) on GPU and on the pod's x86 CPU
cd "$(dirname "$0")/../.."
OUT=${1:-/root/ev-py-misc-msel/$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT"
PIXI=$(command -v pixi || echo ~/.pixi/bin/pixi)
PY="$PIXI run -e default python -u"
LANES=x-metrics-splitters,x-metrics-search
echo "$(date -u +%FT%TZ) $(hostname) $(git rev-parse --short HEAD 2>/dev/null) out $OUT"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
lscpu | grep -m1 'Model name'
$PIXI install -e default > "$OUT/pixi_install.log" 2>&1 || { echo "PIXI INSTALL FAIL"; tail -5 "$OUT/pixi_install.log"; exit 1; }
export MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-4}
echo "== 1. lane check (GPU == CPU)"
$PIXI run -e default python -u tools/algos_lane_check.py $LANES --out "$OUT/lanecheck" > "$OUT/lanecheck.log" 2>&1
echo "lane check exit $?"; grep -E 'RESULT|AGREE|DISAGREE|NOTHING|FAIL' "$OUT/lanecheck.log" | tail -6
HOST=python/mojolearn/host
arm() {  # arm <gpu|cpu> <before|after> <json>
  local env=(env PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=identical OMP_NUM_THREADS=1)
  [ "$2" = before ] && env+=(MOJOLEARN_MSEL_BEFORE=1)
  if [ "$1" = cpu ]; then
    env+=(MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR=$PWD/$HOST)
    "${env[@]}" $PY tools/identity_break.py --lanes $LANES --repeats 1 --fail-on-refused --json "$3" --require-cpu --require-backend cpu
  else
    "${env[@]}" $PY tools/identity_break.py --lanes $LANES --repeats 1 --fail-on-refused --json "$3" --require-backend cuda
  fi
}
echo "== 2. identity lanes before vs after"
for k in gpu cpu; do
  for s in before after; do arm $k $s "$OUT/id_${k}_${s}.json" > "$OUT/id_${k}_${s}.log" 2>&1; echo "$k $s exit $?"; done
done
$PY - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
ok = True
for k in ("gpu", "cpu"):
    try:
        b = json.load(open(f"{out}/id_{k}_before.json"))["cells"]
        a = json.load(open(f"{out}/id_{k}_after.json"))["cells"]
    except Exception as e:
        print(f"{k}: MISSING {e}"); ok = False; continue
    keys = sorted(set(b) | set(a))
    same = [c for c in keys if c in a and c in b and a[c].get("hashes") and a[c].get("hashes") == b[c].get("hashes")]
    for c in keys:
        if c not in same:
            ok = False
            print(f"{k} {c}: before {b.get(c, {}).get('verdict')} after {a.get(c, {}).get('verdict')} DIFFER")
    print(f"{k}: {len(same)} of {len(keys)} cells carry identical hashes before and after")
print("BEFORE==AFTER", "PASS" if ok else "FAIL")
PY
echo "== 3. equality script"
env PYTHONPATH=python $PY tools/py_misc_msel/check.py equal > "$OUT/equal_gpu.log" 2>&1; echo "gpu exit $?"; tail -3 "$OUT/equal_gpu.log"
env PYTHONPATH=python MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR=$PWD/$HOST $PY tools/py_misc_msel/check.py equal > "$OUT/equal_cpu.log" 2>&1; echo "cpu exit $?"; tail -3 "$OUT/equal_cpu.log"
echo "== 4. timing 1M rows"
env PYTHONPATH=python $PY tools/py_misc_msel/check.py time > "$OUT/time_gpu.log" 2>&1; echo "gpu exit $?"; cat "$OUT/time_gpu.log" | tail -20
env PYTHONPATH=python MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR=$PWD/$HOST $PY tools/py_misc_msel/check.py time > "$OUT/time_cpu.log" 2>&1; echo "cpu exit $?"; tail -20 "$OUT/time_cpu.log"
echo "JOB END $(date -u +%FT%TZ)"
