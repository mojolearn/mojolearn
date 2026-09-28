#!/bin/bash
# The light check on one box (2026-09-28): build once, then the GPU column against the
# CPU column in batches of lanes, only the lanes in $LANES (comma separated; the caller
# selects them with tools/lane_select.py --changed-since <last verified commit>).
#   light_job.sh <out dir>
# Environment: LANES (required), BATCH (20), REFERENCE (CPU column JSONs of this commit,
# comma separated; when given, no CPU arm runs here), KEEP_PAR=1 to run par-* lanes.
cd "$(dirname "$0")/../.."
OUT=${1:?out}; mkdir -p "$OUT"
[ -n "${LANES:-}" ] || { echo "LIGHT JOB: LANES is empty; refusing to run every lane"; exit 2; }
export MOJOLEARN_BUILD_LOCK_HELD=1 MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-2}
PIXI=$(command -v pixi || echo ~/.pixi/bin/pixi)
P="$PIXI run -e default python -u tools/merged_check/merged_check.py"
T0=$(date +%s)
echo "$(date -u +%FT%TZ) $(hostname) $(git rev-parse --short HEAD) light out $OUT"
$PIXI install -e default > "$OUT/pixi_install.log" 2>&1 || { echo "PIXI INSTALL FAIL"; tail -5 "$OUT/pixi_install.log"; exit 1; }
[ -f "$OUT/plan.json" ] || $P plan --out "$OUT"
python3 - "$OUT" "$LANES" <<'PY'
import json, sys
out, only = sys.argv[1], [l for l in sys.argv[2].split(",") if l]
p = json.load(open(out + "/plan.json"))
unknown = [l for l in only if l not in p["needed"]]
lanes = [l for l in only if l in p["needed"]]
p["bindings"] = sorted(set().union(*[p["needed"][l] for l in lanes])) if lanes else []
p["lanes"] = lanes
json.dump(p, open(out + "/plan_light.json", "w"))
print(f"light: {len(lanes)} lanes, {len(p['bindings'])} bindings" + (f"; NOT EXPOSED: {unknown}" if unknown else ""))
PY
mkdir -p "$OUT/s" && cp "$OUT/plan_light.json" "$OUT/s/plan.json"
T1=$(date +%s)
$P build --out "$OUT/s" --jobs ${BUILD_JOBS:-4} 2>&1 | tail -3
T2=$(date +%s)
$P light --out "$OUT/s" --batch ${BATCH:-20} ${REFERENCE:+--reference "$REFERENCE"} $([ "${KEEP_PAR:-0}" = 1 ] || echo --skip-par) 2>&1 \
  | grep -E 'light:|LIGHT RESULT|STALE|REFERENCE REFUSED|batch [0-9]+:|alone|ERROR|DISAGREE|REFUSED|NOTHING'
RC=${PIPESTATUS[0]}
T3=$(date +%s)
echo "SECONDS setup $((T1-T0)) build $((T2-T1)) compare $((T3-T2))"
echo "LIGHT JOB END $(date -u +%FT%TZ) rc=$RC"
exit $RC
