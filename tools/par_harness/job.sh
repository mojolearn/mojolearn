#!/bin/bash
# tools/par_harness/job.sh: THE ONE LIGHT JOB of lane/par-harness
# (docs/lanes/progress/par-harness.md), one NVIDIA queue job on TWO GPU slots.
#   clean     each par-* lane below through tools/algos_lane_check.sh on the
#             base fixture, one lane per call so one failure never hides the
#             rest: on this two-GPU slot every lane is on the DEVICE AXIS (the
#             two-device column under the witness vs the one-device column).
#             Builds only what those lanes load (no host binding: no CPU arm).
#   sabotage  par-arima with tools/par_harness/sabotage_par_read_shift.patch:
#             AGREE, then DISAGREE under the patch (every owner above rank 0
#             reads one series early, inert at one device), then AGREE after
#             `git apply -R`.
#   tests     tools/test_algos_lane_check_par.py (the verdict logic and the
#             1-GPU NOT APPLICABLE path).
# PHASES=a,b runs a subset.
set -u
T=$(cd "$(dirname "$0")/../.." && pwd)
EV=${EV:-/root/ev-par-harness/$(date -u +%m%d-%H%M)}
PHASES=${PHASES:-clean,sabotage,tests}
has() { case ",$PHASES," in *",$1,"*) return 0 ;; esac; return 1; }
mkdir -p "$EV"
export MOJOLEARN_BUILD_LOCK_HELD=1 MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-6}
export MOJOLEARN_LANE_CHECK_ARM_TIMEOUT=${MOJOLEARN_LANE_CHECK_ARM_TIMEOUT:-1800}
cd "$T" || exit 1
LANES=${LANES:-par-boosting-clf par-border-types par-byte-lm-offload par-cd-elasticnet par-dbscan par-gmm par-gram-pca par-graph-spectral par-kernel-ridge par-ordered-rmse}
echo "$(date -u +%FT%TZ) $(hostname) out $EV; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}; head $(git rev-parse --short HEAD)"
nvidia-smi -L
PIXI=$(command -v pixi || echo ~/.pixi/bin/pixi)
$PIXI install -e default > "$EV/pixi.log" 2>&1 || { echo "PIXI INSTALL FAIL"; tail -5 "$EV/pixi.log"; exit 1; }
FAILS=0

if has clean; then
  for lane in $LANES; do
    sh tools/algos_lane_check.sh "$lane" --fixtures base --out "$EV/clean/$lane" > "$EV/clean.$lane.log" 2>&1
    rc=$?
    echo "$(date -u +%FT%TZ) CLEAN $lane exit $rc: $(grep -h '^CLEAN: ' "$EV/clean.$lane.log" | tail -1 | cut -c1-600)"
    grep -h '^RESULT' "$EV/clean.$lane.log" | tail -1 | cut -c1-600
    [ $rc -eq 0 ] || FAILS=$((FAILS + 1))
  done
  # the 1-dev vs 2-dev train hash of every lane, from the columns themselves
  $PIXI run -e default python - "$EV/clean" <<'PY'
import json, sys
from pathlib import Path
for d in sorted(Path(sys.argv[1]).iterdir()):
    two, one, w = d / f"clean.{d.name}.gpu.json", d / f"clean.{d.name}.cpu.json", d / f"clean.{d.name}.gpu.json.witness.json"
    if not (two.is_file() and one.is_file()):
        print(f"HASHES {d.name}: columns missing"); continue
    c2, c1 = json.loads(two.read_text())["cells"], json.loads(one.read_text())["cells"]
    wc = json.loads(w.read_text())["cells"] if w.is_file() else {}
    for k in sorted(set(c2) | set(c1)):
        h2 = (c2.get(k) or {}).get("hashes") or [(c2.get(k) or {}).get("verdict")]
        h1 = (c1.get(k) or {}).get("hashes") or [(c1.get(k) or {}).get("verdict")]
        ws = wc.get(k, {})
        print(f"HASHES {k}: one-device {h1[0]} two-device {h2[0]} "
              f"{'EQUAL' if h1[0] == h2[0] else 'DIFFER'}; witness "
              f"{'refused: ' + str(ws.get('refusal'))[:200] if ws.get('refusal') else ws.get('witness', 'MISSING')}")
PY
fi

if has sabotage; then
  sh tools/algos_lane_check.sh par-arima --fixtures base --sabotage tools/par_harness/sabotage_par_read_shift.patch \
     --out "$EV/sabotage" > "$EV/sabotage.log" 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) SABOTAGE par-arima exit $rc"
  grep -h '^CLEAN: \|^SABOTAGED: \|^RESTORED: \|^RESULT' "$EV/sabotage.log" | cut -c1-700
  [ $rc -eq 0 ] || FAILS=$((FAILS + 1))
  git diff --quiet -- python/mojolearn/_parallel_pool.py || { echo "TREE STILL SABOTAGED"; FAILS=$((FAILS + 1)); }
fi

if has tests; then
  $PIXI run -e default python -u tools/test_algos_lane_check_par.py > "$EV/tests.log" 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) TESTS exit $rc: $(tail -1 "$EV/tests.log")"
  [ $rc -eq 0 ] || { tail -30 "$EV/tests.log"; FAILS=$((FAILS + 1)); }
fi

echo "$(date -u +%FT%TZ) DONE: $FAILS failing step(s); evidence $EV"
[ $FAILS -eq 0 ]
