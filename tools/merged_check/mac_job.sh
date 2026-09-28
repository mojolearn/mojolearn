#!/bin/bash
# lane/apple-merged: one Mac's column (Metal GPU arm vs Mac CPU arm) for shard $1 (i/n)
# of every exposed lane; results persist in $2 so a resubmission resumes.
cd "$(dirname "$0")/../.."
SHARD=${1:-0/1}; OUT=${2:?out}; mkdir -p "$OUT"
export MOJOLEARN_BUILD_LOCK_HELD=1 MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-2}
PIXI=$(command -v pixi || echo ~/.pixi/bin/pixi)
P="$PIXI run -e default python -u tools/merged_check/merged_check.py"
echo "$(date -u +%FT%TZ) $(hostname) $(git rev-parse --short HEAD) shard $SHARD out $OUT"
$PIXI install -e default > "$OUT/pixi_install.log" 2>&1 || { echo "PIXI INSTALL FAIL"; tail -5 "$OUT/pixi_install.log"; exit 1; }
[ -f "$OUT/plan.json" ] || $P plan --out "$OUT"
python3 - "$OUT" "$SHARD" "${LANES:-}" <<'PY'
import json, sys
out, (i, n), only = sys.argv[1], (int(x) for x in sys.argv[2].split("/")), sys.argv[3]
p = json.load(open(out + "/plan.json"))
lanes = [l for l in only.split(",") if l] if only else p["lanes"][i::n]
p["bindings"] = sorted(set().union(*[p["needed"][l] for l in lanes]))
p["lanes"] = lanes
json.dump(p, open(out + "/plan_shard.json", "w"))
print(f"shard {i}/{n}: {len(lanes)} lanes, {len(p['bindings'])} bindings")
PY
mkdir -p "$OUT/s" && cp "$OUT/plan_shard.json" "$OUT/s/plan.json"
$P build --out "$OUT/s" --jobs ${BUILD_JOBS:-4} 2>&1 | tail -3
$P clean --out "$OUT/s" --shard 0/1 --cpu-threads default 2>&1 | grep -E 'RESULT|STALE|not AGREE|ERROR|DISAGREE|REFUSED|clean shard'
echo "JOB END $(date -u +%FT%TZ)"
exit 0
