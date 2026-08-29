#!/usr/bin/env bash
# The FAST-tier ExtraTrees timing breakdown, per phase and (where the image
# has rocprof) per kernel, at the higgs speed-arm shape.
#
#   bash tools/et_profile_leg.sh                # on a rented box, via
#                                               # E2_EXTRA_CHECKS=et-profile
#   ET_PROFILE_ROWS=20000 ET_PROFILE_TREES=4 ET_PROFILE_ARMS=128 \
#       bash tools/et_profile_leg.sh            # a local smoke at nice 19
#
# WHY. BOARD_2026-08-28_three-vendor.md section 2.3: the same source and the
# same FAST tier fit ExtraTrees on higgs 1M in 4160 ms on an H100 and in
# 18294 ms on an MI325X, while the random forest and isolation forest lanes
# are FASTER on the MI325X. Nothing had been profiled; the number was a
# whole-fit wall time. This runs `extratrees/bench/fit_once.mojo` -- ONE
# `train_forest_classification_device` call, the shipping merged path -- with
# its `PhaseClock` (range / score / reduce / partition / leaf / host), at
# `DEVICE_TPB` 128 (shipped) beside 256 and 512 (`-D MOJOLEARN_ET_TPB_*`),
# and wraps the shipped arm in `rocprofv3 --kernel-trace --stats` when the
# image carries it. Everything lands under the latest
# bench/results/e1/<stamp>/lanes/et_profile/ so the leg's fetch brings it home.
#
# THE DATA IS HIGGS WHEN IT CAN BE. The AMD speed number was taken on
# higgs-1000000x28 and higgs-2000000x28 (bench/results/fast_speed/
# 2026-08-28-AMD-forest-higgs.md); a breakdown on a synthetic fixture would be
# a breakdown of a different problem. The first 2M rows of HIGGS.csv.gz are
# enough for both rungs (the speed arm takes training rows from the FRONT),
# so only that prefix is parsed. If the download fails the script falls back
# to a synthetic 2M x 28 binary fixture and SAYS SO in every header line.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PATH="$HOME/.pixi/bin:$PATH"

ROWS="${ET_PROFILE_ROWS:-1000000}"          # first rung
ROWS2="${ET_PROFILE_ROWS2:-2000000}"        # second rung (0 = skip)
TREES="${ET_PROFILE_TREES:-100}"            # lane_config: n_estimators 100
DEPTH="${ET_PROFILE_DEPTH:-16}"             # lane_config: max_depth 16
ARMS="${ET_PROFILE_ARMS:-128 256 512}"      # DEVICE_TPB arms
NFEAT=28
NCLASS=2
TOTAL_ROWS=$(( ROWS2 > ROWS ? ROWS2 : ROWS ))

OUT="${ET_PROFILE_OUT:-$(ls -td bench/results/e1/*/ 2>/dev/null | head -1)}"
[ -n "$OUT" ] || OUT="bench/results/e1/et_profile_local"
OUT="${OUT%/}/lanes/et_profile"
mkdir -p "$OUT" build
DATA="${ET_PROFILE_DATA:-/root/et_profile_data}"
mkdir -p "$DATA"
PY="$REPO/.pixi/envs/default/bin/python"
[ -x "$PY" ] || PY=python3
# coreutils `timeout` on the box; the launching Mac has none, and a local
# smoke runs unbounded rather than not at all.
TO() { if command -v timeout >/dev/null 2>&1; then timeout -k 30 "$@"; else shift; "$@"; fi; }

echo "et_profile: commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown) rows=$ROWS/$ROWS2 trees=$TREES depth=$DEPTH arms=$ARMS"
echo "et_profile: out $OUT"
uname -a
{ rocm-smi --showproductname 2>/dev/null | grep -i "card series\|name" | head -2; } || true
ROCPROF=""
for c in rocprofv3 /opt/rocm/bin/rocprofv3; do
  if command -v "$c" >/dev/null 2>&1; then ROCPROF="$c"; break; fi
done
echo "et_profile: rocprofv3 = ${ROCPROF:-NOT FOUND} ; rocprof = $(command -v rocprof || echo NOT FOUND)"
ls /opt/rocm/bin 2>/dev/null | grep -i prof | tr '\n' ' '; echo

# ---- data: higgs prefix, or synthetic ------------------------------------
NAME=higgs
if [ ! -s "$DATA/higgs_y.f32" ] || [ ! -s "$DATA/higgs_Xcol.f32" ]; then
  GZ=""
  for cand in /root/datasets/gbm-bench/higgs/HIGGS.csv.gz "$DATA/HIGGS.csv.gz"; do
    [ -s "$cand" ] && GZ="$cand" && break
  done
  if [ -z "$GZ" ] && [ "${ET_PROFILE_NO_DOWNLOAD:-0}" != 1 ]; then
    echo "et_profile: downloading HIGGS.csv.gz (2.6 GB, bounded 900 s) $(date +%T)"
    if TO 900 curl -sSL --retry 3 -o "$DATA/HIGGS.csv.gz" \
         https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz; then
      GZ="$DATA/HIGGS.csv.gz"
    else
      echo "et_profile: download FAILED $(date +%T)"; rm -f "$DATA/HIGGS.csv.gz"
    fi
  fi
  if [ -n "$GZ" ]; then
    echo "et_profile: parsing the first $TOTAL_ROWS rows of $GZ $(date +%T)"
    "$PY" - "$GZ" "$DATA" "$TOTAL_ROWS" <<'PY' || NAME=synth
import sys, numpy as np
gz, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
a = np.loadtxt(gz, delimiter=",", dtype=np.float32, max_rows=n)
assert a.shape == (n, 29), a.shape
y = np.ascontiguousarray(a[:, 0]); x = a[:, 1:]
np.ascontiguousarray(x.T).tofile(out + "/higgs_Xcol.f32")   # column-major
y.tofile(out + "/higgs_y.f32")
print("higgs prefix written:", x.shape, "positives", float(y.mean()))
PY
  else
    NAME=synth
  fi
fi
if [ "$NAME" = synth ] && { [ ! -s "$DATA/synth_y.f32" ] || [ ! -s "$DATA/synth_Xcol.f32" ]; }; then
  echo "et_profile: SYNTHETIC fixture ${TOTAL_ROWS}x$NFEAT (higgs unavailable) $(date +%T)"
  "$PY" - "$DATA" "$TOTAL_ROWS" "$NFEAT" <<'PY'
import sys, numpy as np
out, n, d = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
r = np.random.default_rng(0x0F17)
x = r.standard_normal((n, d), dtype=np.float32)
w = r.standard_normal(d).astype(np.float32)
y = ((x @ w + 0.5 * r.standard_normal(n).astype(np.float32)) > 0).astype(np.float32)
np.ascontiguousarray(x.T).tofile(out + "/synth_Xcol.f32")
y.tofile(out + "/synth_y.f32")
print("synthetic written:", x.shape, "positives", float(y.mean()))
PY
fi
echo "et_profile: dataset=$NAME total_rows=$TOTAL_ROWS"

# ---- the arms -------------------------------------------------------------
for tpb in $ARMS; do
  DEF=""
  [ "$tpb" != 128 ] && DEF="-D MOJOLEARN_ET_TPB_$tpb=1"
  echo "et_profile: build tpb=$tpb $(date +%T)"
  if ! ${NICE:-} pixi run mojo build -I . $DEF extratrees/bench/fit_once.mojo \
        -o "build/et_fit_once_tpb$tpb" > "$OUT/build_tpb$tpb.log" 2>&1; then
    echo "et_profile: BUILD FAILED tpb=$tpb"; grep -m3 "error" "$OUT/build_tpb$tpb.log"; continue
  fi
done

run_arm() {  # tpb rows phases?
  local tpb=$1 rows=$2 ph=${3:-}
  local bin="build/et_fit_once_tpb$tpb"
  [ -x "$bin" ] || { echo "ET-PROFILE tpb=$tpb rows=$rows NO BINARY"; return; }
  echo "--- ET-PROFILE dataset=$NAME tpb=$tpb rows=$rows trees=$TREES depth=$DEPTH ${ph:+(phases)} $(date +%T)"
  TO "${ET_PROFILE_ARM_TIMEOUT:-600}" ${NICE:-} "$bin" "$DATA" "$NAME" "$rows" "$NFEAT" "$NCLASS" \
      "$TREES" "$DEPTH" sqrt $ph 2>&1 | tee "$OUT/fit_${NAME}_${rows}_tpb${tpb}${ph:+_phases}.txt"
}

for tpb in $ARMS; do
  run_arm "$tpb" "$ROWS" phases
done
if [ "$ROWS2" -gt 0 ]; then
  for tpb in $ARMS; do
    run_arm "$tpb" "$ROWS2"
  done
fi

# ---- per-kernel device time, shipped arm and the widest arm ---------------
if [ -n "$ROCPROF" ]; then
  for tpb in $(echo "$ARMS" | awk '{print $1; if (NF>1) print $NF}'); do
    bin="build/et_fit_once_tpb$tpb"
    [ -x "$bin" ] || continue
    echo "--- rocprofv3 tpb=$tpb rows=$ROWS $(date +%T)"
    rm -rf "$OUT/rocprof_tpb$tpb"
    TO "${ET_PROFILE_PROF_TIMEOUT:-900}" "$ROCPROF" --kernel-trace --stats \
        -d "$OUT/rocprof_tpb$tpb" -o "et_tpb$tpb" --output-format csv -- \
        "$bin" "$DATA" "$NAME" "$ROWS" "$NFEAT" "$NCLASS" "$TREES" "$DEPTH" sqrt \
        > "$OUT/rocprof_tpb$tpb.log" 2>&1 || echo "rocprofv3 exit $? (see rocprof_tpb$tpb.log)"
    tail -3 "$OUT/rocprof_tpb$tpb.log"
    stats=$(find "$OUT/rocprof_tpb$tpb" -name '*kernel_stats.csv' | head -1)
    if [ -n "$stats" ]; then
      echo "kernel stats (top 12 by total ns):"
      "$PY" - "$stats" <<'PY'
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1])))
key = "TotalDurationNs" if rows and "TotalDurationNs" in rows[0] else list(rows[0].keys())[2]
rows.sort(key=lambda r: -float(r[key]))
tot = sum(float(r[key]) for r in rows)
for r in rows[:12]:
    print("  %6.1f%%  %9.1f ms  calls %8s  %s" % (100*float(r[key])/tot, float(r[key])/1e6, r.get("Calls",""), r["Name"][:90]))
print("  total device kernel time %.1f ms over %d kernels" % (tot/1e6, len(rows)))
PY
    else
      echo "no kernel_stats.csv under $OUT/rocprof_tpb$tpb"
    fi
    # the trace itself is large; keep only the stats
    find "$OUT/rocprof_tpb$tpb" -name '*kernel_trace.csv' -size +20M -delete 2>/dev/null
  done
fi

# ---- identical-tier reach, when phase 9 has built the identical trees .so --
if [ "${ET_PROFILE_SKIP_BREAK:-0}" != 1 ] && ls python/mojolearn/identical/_mojolearn_trees*.so >/dev/null 2>&1; then
  echo "--- identity_break et-clf,et-reg (identical) $(date +%T)"
  MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$REPO/python" \
    TO 900 pixi run -e gbmbench python3 -u tools/identity_break.py \
      --lanes et-clf,et-reg --vendor "${MOJOLEARN_P9_VENDOR:-$(hostname)}" \
      --json "$OUT/identity_break.et.identical.json" \
      > "$OUT/identity_break.et.identical.txt" 2>&1 || echo "identity_break exit $?"
  tail -6 "$OUT/identity_break.et.identical.txt"
fi
echo "et_profile: done $(date +%T)"
