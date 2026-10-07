#!/usr/bin/env bash
# six_lane_grid_decide.sh <run-dir-nvidia> <run-dir-amd> [--out DIR] [--dry-run]
#
# After tools/six_lane_grid_run.sh finished on both boxes and both run directories are in one place: IDENTICAL is
# decided by NVIDIA and AMD together (CLAUDE.md), so floors, verdicts, identity and the per-switch decisions need
# both vendors' receipts. Steps (each writes into --out, default <parent of run-dir-nvidia>/grid-decide):
#   1 floors     six_lane_timing.py floors --from-pairs   (incumbent arm B repeated across cells; no A/A pass)
#   2 verdicts   six_lane_timing.py verdicts              (FASTER/SLOWER only beyond both vendors' floors, same direction)
#   3 manifest   six_lane_grid_run.py manifest            (one case per configuration x workload, nvidia-native + amd)
#   4 compare    six_lane_compare_results.py --manifest   (NVIDIA == AMD output bits per arm; new dir per invocation)
#   5 quality    six_lane_grid_run.py quality             (candidate A vs incumbent B, tools/af_quality.py)
#   6 decide     six_lane_grid_decide.py                  (PROMOTE / SPLIT / DELETE / HOLD / NOT_MEASURED per switch arm)
# Re-running is safe: steps 1-3 and 5-6 rewrite their outputs from the receipts; step 4 writes compare-<UTC stamp>.
set -uo pipefail
[ $# -ge 2 ] || { sed -n '2p' "$0" | sed 's/^# //'; exit 2; }
NV=$1; AMD=$2; shift 2
OUT=; DRY=0
while [ $# -gt 0 ]; do
  case $1 in --out) OUT=$2; shift 2;; --dry-run) DRY=1; shift;; *) echo "unknown option: $1" >&2; exit 2;; esac
done
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$(dirname "$NV")/grid-decide}
PY=$(command -v python3)
T=$REPO/tools
PROBLEMS=0
problem() { echo "grid-decide: PROBLEM: $*" >&2; PROBLEMS=$((PROBLEMS + 1)); }
for d in "$NV" "$AMD"; do
  [ -d "$d/results" ] || problem "$d/results missing (run tools/six_lane_grid_run.sh on that box first)"
  [ -f "$d/grid/grid-matrix.json.gz" ] || problem "$d/grid/grid-matrix.json.gz missing"
done
if [ -f "$NV/grid/grid-matrix.json.gz" ] && [ -f "$AMD/grid/grid-matrix.json.gz" ]; then
  a=$(cksum < "$NV/grid/grid-matrix.json.gz"); b=$(cksum < "$AMD/grid/grid-matrix.json.gz")
  [ "$a" = "$b" ] || problem "the two run dirs were planned from different grids (different freezes?)"
  for d in "$NV" "$AMD"; do [ -f "$d/grid/.source" ] && echo "grid-decide: $(basename "$d") grid at $(cut -c1-12 "$d/grid/.source")"; done
fi
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
step() {  # step <name> <argv...>
  local name=$1; shift
  if [ $DRY = 1 ]; then printf '  $'; printf ' %q' "$@"; printf '  > %s\n' "$OUT/logs/$name.log"; return 0; fi
  [ $PROBLEMS = 0 ] || { echo "grid-decide: $name skipped after an earlier problem"; return 0; }
  "$@" > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  echo "grid-decide: $name rc=$rc $(tail -1 "$OUT/logs/$name.log" | cut -c1-240)"
  # compare exits 1 on a MISMATCH and 2 on incomplete cases: both are results, not tool failures.
  case $name:$rc in *:0|compare:1|compare:2) ;; *) problem "$name failed (rc=$rc); see $OUT/logs/$name.log";; esac
}
[ $DRY = 1 ] || { [ $PROBLEMS = 0 ] || exit 1; mkdir -p "$OUT/logs"; }
step floors "$PY" "$T/six_lane_timing.py" floors --from-pairs "$NV/results" "$AMD/results" --out "$OUT/floors.json"
step verdicts "$PY" "$T/six_lane_timing.py" verdicts "$NV/results" "$AMD/results" --floors "$OUT/floors.json" --out "$OUT/verdicts.json"
step manifest "$PY" "$T/six_lane_grid_run.py" manifest --nvidia "$NV/results" --amd "$AMD/results" --out "$OUT/compare-input.json"
step compare "$PY" "$T/six_lane_compare_results.py" --manifest "$OUT/compare-input.json" --aa-floors "$OUT/floors.json" --out "$OUT/compare-$STAMP"
step quality "$PY" "$T/six_lane_grid_run.py" quality "$NV/results" "$AMD/results" --out "$OUT/quality.json"
step decide "$PY" "$T/six_lane_grid_decide.py" --matrix "$NV/grid/grid-matrix.json.gz" --verdicts "$OUT/verdicts.json" \
  --identity "$OUT/compare-$STAMP/report.json" --quality "$OUT/quality.json" --out "$OUT/decisions"
if [ $DRY = 1 ]; then echo "grid-decide: dry run: $PROBLEMS problem(s); nothing was run"; exit $(( PROBLEMS > 0 )); fi
echo "grid-decide: outputs in $OUT (decisions/GRID_DECISIONS.md, decisions/grid-decisions.json); $PROBLEMS problem(s)"
exit $(( PROBLEMS > 0 ))
