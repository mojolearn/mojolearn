#!/bin/bash
# lane/linfit-speed NVIDIA job: build this tree's (and the base tree's)
# IDENTICAL x_linear binding, stage the board's blocks, gate, then time.
#   STAGE=gate  small shape: new vs base vs GLM team arm vs tiny launches (bits)
#   STAGE=full  the board's shapes: new and base (bits + fit seconds)
set -euo pipefail
LANE_DIR=/root/mojolearn-linfit-speed
BASE_DIR=${BASE_DIR:-/root/mojolearn-linfit-speed-base}
DATA=${DATA:-/root/linfit-data}
OUT=$LANE_DIR/bench/results/linfit_speed/out-$(date -u +%Y%m%dT%H%M%SZ)-${STAGE:-gate}
mkdir -p "$OUT"
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=${MOJOLEARN_GPU_ARCHS:-sm_89}
PY="pixi run -e default python"
for t in "$LANE_DIR" "$BASE_DIR"; do
  (cd "$t" && git log --oneline -1 2>/dev/null || true; cd "$t" && sh bindings/build_x_linear.sh) 2>&1 | grep -A8 " error:\|^built"
done
if [ ! -f "$DATA/reg-istella.npz" ] || [ ! -f "$DATA/cls-istella.npz" ]; then
  (cd "$LANE_DIR" && $PY tools/bench_board_algos.py prep --data "$DATA" --lanes sgd-clf,sgd-reg,poisson --datasets taxi,istella) 2>&1 | tail -5
fi
nvidia-smi --query-gpu=name --format=csv,noheader | tee "$OUT/gpu.txt"
run() { # tree tag args...
  local t=$1 tag=$2; shift 2
  (cd "$t" && $PY tools/linfit_speed.py --data "$DATA" --out "$OUT/$tag.json" "$@") 2>&1 | tee "$OUT/$tag.log" | grep -v '^\s*$' | tail -12
}
cp "$LANE_DIR/tools/linfit_speed.py" "$BASE_DIR/tools/linfit_speed.py"
if [ "${STAGE:-gate}" = gate ]; then
  G="--rows ${GATE_ROWS:-100000} --max-iter ${GATE_ITER:-5}"
  run "$LANE_DIR" new $G --env MOJOLEARN_X_LINEAR_GW_TRACE=1
  run "$LANE_DIR" new-tiny $G --lanes poisson --env MOJOLEARN_X_LINEAR_GW_STEPS=4096
  run "$LANE_DIR" new-team $G --lanes poisson --env MOJOLEARN_X_LINEAR_GLM_TEAM=1
  run "$BASE_DIR" base $G
elif [ "${STAGE}" = probe ]; then
  # where SGD's per-row time goes: the shuffle pipeline (shuffle off), the chain (d)
  P="--max-iter ${PROBE_ITER:-3} --lanes sgd-reg"
  run "$LANE_DIR" probe-new $P
  run "$LANE_DIR" probe-new-noshuf $P --set shuffle=False
  run "$BASE_DIR" probe-base $P
  run "$BASE_DIR" probe-base-noshuf $P --set shuffle=False
else
  run "$LANE_DIR" new --lanes "${LANES:-poisson,sgd-reg,sgd-clf}" --env MOJOLEARN_X_LINEAR_GW_TRACE=1
  [ "${SKIP_BASE:-0}" = 1 ] || run "$BASE_DIR" base --lanes "${LANES:-poisson,sgd-reg,sgd-clf}"
fi
python3 - "$OUT" <<'PY'
import json, glob, os, sys
out = sys.argv[1]
recs = {}
for f in sorted(glob.glob(os.path.join(out, "*.json"))):
    for r in json.load(open(f)):
        recs.setdefault((r["lane"], r["dataset"]), {})[os.path.basename(f)[:-5]] = r
bad = 0
for k, arms in sorted(recs.items()):
    shas = {a: r["sha"] for a, r in arms.items()}
    same = len(set(shas.values())) == 1
    bad += not same
    print("%-8s %-8s %s  %s" % (k[0], k[1], "BITS-EQUAL" if same else "BITS-DIFFER",
          "  ".join("%s=%ss(%s,it%s)" % (a, r["fit_s"], r["sha"], r["n_iter"]) for a, r in sorted(arms.items()))))
print("VERDICT", "PASS" if bad == 0 else "FAIL (%d differ)" % bad)
PY
