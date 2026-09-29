#!/bin/sh
# The NEW code's proof on the M3 Ultra (algos and neural half) (NVIDIA values, new lanes): every race of every family at a small row cap,
# one round, into its OWN output directory (never the real board's). Its only
# question: does every race print a BOARD-PARAMS line reading MATCHED?
set -u
OUT="$HOME/mojolearn-evidence/bench-board/2026-09-29_m3ultra_proof2"
CACHE="$HOME/bench-board-cache"
PY="$HOME/mojolearn/.pixi/envs/default/bin/python3.13"
mkdir -p "$OUT"
"$PY" tools/bench_board.py --vendor apple --mojolearn-version 0.8.25 \
    --base-python "$PY" --out "$OUT" --cache "$CACHE" --no-cpu-arm \
    --rows 2000 --rounds 1 --neural-shape small --no-infer --skip-failed "$@" > "$OUT/run.log" 2>&1
echo "board exit $?"
"$PY" - "$OUT/board.json" <<'PY'
import json, sys, collections
b = json.load(open(sys.argv[1]))
c = collections.Counter()
bad = []
for k, r in b.get("races", {}).items():
    v = r.get("params_check") or "NOT CHECKED"
    c[(r.get("family"), v if v in ("MATCHED", "REFUSED") else ("NOT CHECKED" if "NOT" in str(v) else str(v)[:30]))] += 1
    if v != "MATCHED":
        bad.append("%s: %s" % (k, str(v)[:200]))
for k, n in sorted(c.items(), key=str):
    print("PROOF", k[0], k[1], n)
print("PROOF-NOT-MATCHED", len(bad))
for line in bad:
    print("  ", line)
PY
