#!/bin/bash
set -u
PY=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python
export PYTHONPATH=/Users/andrewhendel/mojolearn-wt/gpt3-tooling/python
cd /Users/andrewhendel/mojolearn-wt/gpt3-tooling
O=<scratchpad>/live; mkdir -p $O
COL=bench/results/par_lm_xvendor/2026-09-21/apple-m4-1gpu.json
echo "=== gathered, 2 local workers (expect recorded column)"
$PY tools/live_xvendor.py local --workers 2 --port 7793 --out $O/gathered.json --expect $COL 2>&1 | tail -2
echo "=== chained, 2 local workers (expect recorded column)"
$PY tools/live_xvendor.py local --workers 2 --port 7794 --chained --out $O/chained2.json --expect $COL 2>&1 | tail -2
echo "=== chained, 3 local workers, unequal blocks (expect recorded column)"
$PY tools/live_xvendor.py local --workers 3 --port 7795 --chained --out $O/chained3.json --expect $COL 2>&1 | tail -2
$PY - <<PY
import json
g=json.load(open("$O/gathered.json")); c2=json.load(open("$O/chained2.json")); c3=json.load(open("$O/chained3.json"))
print("splits:", c2["split"], c3["split"])
print("state hashes equal across gathered/chained2/chained3:", [r["state"] for r in g["rows"]]==[r["state"] for r in c2["rows"]]==[r["state"] for r in c3["rows"]])
print("total hashes equal:", [r["total_sha256"] for r in g["rows"]]==[r["total_sha256"] for r in c2["rows"]]==[r["total_sha256"] for r in c3["rows"]])
PY
