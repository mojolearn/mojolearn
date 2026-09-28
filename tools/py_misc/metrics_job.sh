#!/bin/bash
# lane py-misc-metrics: ONE light job on a shared NVIDIA pod.
#  1. tools/algos_lane_check.sh on the x-metrics lanes (stamped builds, GPU
#     vs CPU AGREE) with the native epilogues (the default route);
#  2. the same harness, both columns, with MOJOLEARN_METRICS_EPILOGUE=python
#     (the Python reference route) in the SAME build;
#  3. native vs python digests per column, cell for cell;
#  4. before/after timing of each moved epilogue at 1M (GPU binding, then
#     the x86 host binding), tools/py_misc/metrics_time.py.
cd "$(dirname "$0")/../.."
OUT=${1:-/root/ev-py-misc-metrics/run}; mkdir -p "$OUT"
LANES=x-metrics-ranking,x-metrics-cluster,x-metrics-classification
export PATH=$HOME/.pixi/bin:$PATH MOJOLEARN_BUILD_LOCK_HELD=1 MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-2}
echo "$(date -u +%FT%TZ) $(hostname) $(git rev-parse --short HEAD) out $OUT"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
lscpu | grep 'Model name' | head -1
pixi run -e default sh tools/algos_lane_check.sh $LANES --out "$OUT/lc" > "$OUT/lane_check.log" 2>&1
echo "lane_check exit $?"; grep -E 'RESULT|AGREE|DISAGREE|NOTHING|FAIL' "$OUT/lane_check.log" | tail -8
PY=".pixi/envs/default/bin/python"
H=tools/identity_break.py
for kind in gpu cpu; do
  for lane in ${LANES//,/ }; do
    if [ $kind = cpu ]; then
      ENVS="MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR=$PWD/python/mojolearn/host"; FLAGS="--require-cpu --require-backend cpu"
    else
      ENVS=""; FLAGS="--require-backend cuda"
    fi
    env PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=identical OMP_NUM_THREADS=1 MOJOLEARN_METRICS_EPILOGUE=python $ENVS \
      $PY -u $H --lanes $lane --repeats 1 --fail-on-refused --json "$OUT/py.$lane.$kind.json" $FLAGS \
      > "$OUT/py.$lane.$kind.log" 2>&1
    echo "python-route $lane $kind exit $?"
  done
done
$PY - "$OUT" $LANES <<'PY'
import json, sys
out, lanes = sys.argv[1], sys.argv[2].split(",")
bad = 0
for lane in lanes:
    for kind in ("gpu", "cpu"):
        try:
            a = json.load(open(f"{out}/lc/clean.{lane}.{kind}.json"))["cells"]
            b = json.load(open(f"{out}/py.{lane}.{kind}.json"))["cells"]
        except Exception as e:
            print(f"MISSING {lane} {kind}: {e}"); bad += 1; continue
        n = same = 0
        for key in sorted(set(a) | set(b)):
            ha, hb = (a.get(key) or {}).get("hashes"), (b.get(key) or {}).get("hashes")
            n += 1
            if ha is not None and ha == hb:
                same += 1
            else:
                print(f"  DIFFER {kind} {key}"); bad += 1
            for part in ("infer", "model", "batch"):
                pa, pb = (a.get(key) or {}).get(part), (b.get(key) or {}).get(part)
                if pa != pb:
                    print(f"  DIFFER {kind} {key} part {part}"); bad += 1
        print(f"native vs python {lane} {kind}: {same}/{n} cells equal")
print("NATIVE==PYTHON" if not bad else f"NATIVE!=PYTHON ({bad})")
PY
for kind in gpu cpu; do
  if [ $kind = cpu ]; then ENVS="MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR=$PWD/python/mojolearn/host"; else ENVS=""; fi
  env PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=identical OMP_NUM_THREADS=1 $ENVS $PY -u tools/py_misc/metrics_time.py $kind \
    > "$OUT/time.$kind.log" 2>&1
  echo "timing $kind exit $?"; cat "$OUT/time.$kind.log" | tail -30
done
echo "JOB END $(date -u +%FT%TZ)"
