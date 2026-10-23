#!/usr/bin/env bash
# The sm_120 tree-identity probe (lane sm120-tree-identity, 2026-10-23).
#
# Finds the reader behind the RTX 5090 tree findings
# (~/mojolearn-evidence/audit-20261023/sm120-findings.md): et-reg's digest
# moving run to run and its ILLEGAL_ADDRESS after a RandomForest fit, and the
# deterministic sm_120-only digests of the Ordered GBDT lanes and the
# ExtraTrees-regression tree lanes. Runs on any NVIDIA box: the sanitizer
# modes need compute-sanitizer >= the box's arch (12.4 is enough on an L40S
# sm_89; an sm_120 box needs >= 12.8).
#
#   bash tools/sm120_tree_identity_probe.sh initcheck [lanes]   uninitialized global reads
#   bash tools/sm120_tree_identity_probe.sh racecheck [lanes]   shared-memory hazards
#   bash tools/sm120_tree_identity_probe.sh synccheck [lanes]   barrier / warp-sync misuse
#   bash tools/sm120_tree_identity_probe.sh repeat    [lanes]   each lane in 3 fresh processes, then after rf-reg
#   bash tools/sm120_tree_identity_probe.sh trace     [lanes]   MOJOLEARN_IDENTITY_TRACE per lane (diff across boxes
#                                                               with tools/identity_trace_diff.py)
#
# Output: one directory under $OUT (default ~/sm120_probe/<mode>-<utc>), one
# log per lane, and a grep-sized summary on stdout. Logs are never printed
# whole. The poison arm needs a trees binding built with
# `-D MOJOLEARN_ET_WS_POISON_BYTE=165`; run `repeat` on that build and compare
# its digests with a default build's: a digest that moves is a read-before-write.
set -u
MODE=${1:-}
LANES=${2:-et-reg,trees-et-deviance,trees-dt-random,gbdt-ordered,gbdt-ordered-rmse,gbdt-ordered-bayesian-noise,gbdt-binary-columns,gbdt-bfa-quantile}
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python3}
SAN=${COMPUTE_SANITIZER:-compute-sanitizer}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${OUT:-$HOME/sm120_probe/$MODE-$STAMP}
mkdir -p "$OUT"
cd "$HERE" || exit 2

ib() {  # ib <lanes> <json> -- run identity_break once, CUDA required
  "$PY" tools/identity_break.py --require-backend cuda --lanes "$1" --json "$2"
}

summary() {  # summary <log> -- verdict lines and the first sanitizer findings only
  grep -m 20 -E 'STABLE|MOVED|REFUSED|DIVERGENT|ILLEGAL|Error|ERROR SUMMARY|Uninitialized|Race reported|Barrier error|Invalid __' "$1" | cut -c1-240
}

case "$MODE" in
  initcheck|racecheck|synccheck)
    command -v "$SAN" >/dev/null 2>&1 || { echo "probe: $SAN not found"; exit 2; }
    "$SAN" --version | tail -n 1
    IFS=, read -r -a LIST <<< "$LANES"
    for lane in "${LIST[@]}"; do
      # rf-reg first for et-reg: the reported crash needs a RandomForest fit
      # in the same process before it.
      run="$lane"
      [ "$lane" = "et-reg" ] && run="rf-reg,et-reg"
      log="$OUT/$MODE.$lane.log"
      "$SAN" --tool "$MODE" --target-processes all --print-limit 50 \
        $( [ "$MODE" = initcheck ] && echo "--track-unused-memory no" ) \
        "$PY" tools/identity_break.py --require-backend cuda --lanes "$run" \
        --json "$OUT/$MODE.$lane.json" > "$log" 2>&1
      echo "== $MODE $run rc=$?"
      summary "$log"
      # The kernel names behind the first findings (the reader to fix).
      grep -E '^=+ +at ' "$log" | sort | uniq -c | sort -rn | head -n 10 | cut -c1-240
    done
    ;;
  repeat)
    IFS=, read -r -a LIST <<< "$LANES"
    for lane in "${LIST[@]}"; do
      for i in 1 2 3; do
        log="$OUT/repeat.$lane.$i.log"
        ib "$lane" "$OUT/repeat.$lane.$i.json" > "$log" 2>&1
        echo "== repeat $lane run $i rc=$?"
        summary "$log"
      done
      log="$OUT/repeat.rf-reg+$lane.log"
      ib "rf-reg,$lane" "$OUT/repeat.rf-reg+$lane.json" > "$log" 2>&1
      echo "== repeat rf-reg,$lane rc=$?"
      summary "$log"
    done
    ;;
  trace)
    IFS=, read -r -a LIST <<< "$LANES"
    for lane in "${LIST[@]}"; do
      log="$OUT/trace.$lane.log"
      MOJOLEARN_IDENTITY_TRACE="$OUT/trace.$lane.trace" ib "$lane" "$OUT/trace.$lane.json" > "$log" 2>&1
      echo "== trace $lane rc=$? records=$(wc -l < "$OUT/trace.$lane.trace" 2>/dev/null || echo 0)"
      summary "$log"
    done
    echo "diff across boxes: $PY tools/identity_trace_diff.py <boxA>/trace.<lane>.trace <boxB>/trace.<lane>.trace"
    ;;
  *)
    sed -n '2,25p' "$0"
    exit 2
    ;;
esac
echo "probe: logs in $OUT"
