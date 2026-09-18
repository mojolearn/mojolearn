#!/bin/bash
# lane/sabotage-sweep, shard i: par-scaler and par-scaler-minmax after the
# NumericalMismatch repair. THE UNFIXED SIDE IS ALREADY RECORDED
# (a-linear-neighbors: sabotage cell REFUSED, "transform_scaler and plain
# transform differ: 2847 bytes of 16384"). Two things have to hold here.
#   1. the sabotage cell now carries HASHES that differ, not a refusal
#   2. the CLEAN cell is byte-identical to the one committed before the repair
set -u
O="$LEG_OUT"
LANES='par-scaler,par-scaler-minmax'
st() { n=$1; shift; "$@"; rc=$?; printf '%s\t%s\n' "$n" "$rc" >> "$O/step_exit_codes.txt"; return 0; }
echo "shard i nproc=$(nproc) $(uname -m) $(python3 --version 2>&1)" > "$O/box.txt"
sha256sum python/mojolearn/host/*.so python/mojolearn/host-sabotage/*.so > "$O/so_sha256.txt" 2>&1
python3 - <<'PYEOF' > "$O/arm_bytes_differ.txt" 2>&1
import hashlib, os
p="python/mojolearn/host"; q="python/mojolearn/host-sabotage"
def h(f): return hashlib.sha256(open(f,'rb').read()).hexdigest()
for b in sorted(os.listdir(q)):
    if b.endswith(".so") and os.path.exists(os.path.join(p,b)):
        print(("SAME  " if h(os.path.join(p,b))==h(os.path.join(q,b)) else "DIFFER"), b)
PYEOF
SAB="MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1"
st prod python3 tools/identity_break.py --lanes "$LANES" --fixtures base,ties --repeats 2 --json "$O/cpu-x86.json" > "$O/prod.log" 2>&1
st sab  env $SAB python3 tools/identity_break.py --lanes "$LANES" --fixtures base,ties --repeats 2 --json "$O/cpu-x86.sabotage.json" > "$O/sab.log" 2>&1
st diff python3 tools/identity_break.py --diff "$O/cpu-x86.json" "$O/cpu-x86.sabotage.json" > "$O/diff.clean-vs-sabotage.txt" 2>&1
# The clean cell must not have moved. par-scaler's clean cells are committed in
# the a-linear-neighbors shard, taken before the repair.
st diff_prod python3 tools/identity_break.py --diff \
   bench/results/identity_break/2026-09-17_sabotage-sweep/a-linear-neighbors/cpu-x86.json \
   "$O/cpu-x86.json" --lanes par-scaler > "$O/diff.clean-unmoved.txt" 2>&1
python3 - "$O" <<'PYEOF' > "$O/cells.txt" 2>&1
import json, sys
O = sys.argv[1]
for f in ("cpu-x86", "cpu-x86.sabotage"):
    j = json.load(open(f"{O}/{f}.json"))
    print("==", f, "repeats", j.get("repeats"))
    for k, v in sorted(j["cells"].items()):
        print(f"  {k:28s} {v.get('verdict'):10s} {v.get('hashes')} {str(v.get('error'))[:70]}")
PYEOF
tail -25 "$O/prod.log" > "$O/prod.tail.txt" 2>&1
tail -25 "$O/sab.log" > "$O/sab.tail.txt" 2>&1
