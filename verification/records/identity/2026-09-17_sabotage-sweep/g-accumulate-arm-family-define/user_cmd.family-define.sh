#!/bin/bash
# lane/sabotage-sweep, shard g: the accumulate arm, watched failing.
# The UNFIXED side is already recorded: with no arm on host_samba_accumulate
# the training family's sabotage build left ordered-gradient-sum's `gradients`
# hash unchanged on ALL NINE fixtures
# (bench/results/identity_break/2026-09-17_sabotage-sweep/g-ordered-gradient-sum).
# This run is the same nine fixtures with SAMBA_ACCUMULATE_HOST_SABOTAGE
# compiled in, plus the production check that matters just as much: the
# production column must still equal the pre-change one bit for bit, because
# the arm lives under a comptime if a production build does not compile.
set -u
O="$LEG_OUT"
st() { n=$1; shift; "$@"; rc=$?; printf '%s\t%s\n' "$n" "$rc" >> "$O/step_exit_codes.txt"; return 0; }
echo "shard g-fix nproc=$(nproc) $(uname -m) $(python3 --version 2>&1)" > "$O/box.txt"
sha256sum python/mojolearn/host/*.so python/mojolearn/host-sabotage/*.so > "$O/so_sha256.txt" 2>&1
python3 - <<'PYEOF' > "$O/arm_bytes_differ.txt" 2>&1
import hashlib, os
p="python/mojolearn/host"; q="python/mojolearn/host-sabotage"
def h(f): return hashlib.sha256(open(f,'rb').read()).hexdigest()
for b in sorted(os.listdir(q)):
    if not b.endswith(".so"): continue
    a=os.path.join(p,b)
    if os.path.exists(a):
        print(("SAME  " if h(a)==h(os.path.join(q,b)) else "DIFFER"), b)
PYEOF
grep -c 'SAMBA_ACCUMULATE_HOST_SABOTAGE' training/host/samba_ops_oracle.mojo > "$O/arm_sites.txt" 2>&1
SAB="MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1"
st prod python3 tools/identity_break.py --lanes ordered-gradient-sum --repeats 2 --json "$O/cpu-x86.json" > "$O/prod.log" 2>&1
st sab  env $SAB python3 tools/identity_break.py --lanes ordered-gradient-sum --repeats 2 --json "$O/cpu-x86.sabotage.json" > "$O/sab.log" 2>&1
st diff python3 tools/identity_break.py --diff "$O/cpu-x86.json" "$O/cpu-x86.sabotage.json" > "$O/diff.clean-vs-sabotage.txt" 2>&1
# The production side must not have moved. These five training lanes have a
# committed production column from before the change, in the d-neural shard.
T=cross-entropy-arms,mlp,optim-adam-clip,optim-sgd,training-primitives
st prod_training python3 tools/identity_break.py --lanes "$T" --fixtures base,ties --repeats 2 --json "$O/cpu-x86.training.json" > "$O/prod_training.log" 2>&1
st diff_prod python3 tools/identity_break.py --diff \
   bench/results/identity_break/2026-09-17_sabotage-sweep/d-neural/cpu-x86.json \
   "$O/cpu-x86.training.json" --lanes "$T" > "$O/diff.production-unmoved.txt" 2>&1
tail -20 "$O/prod.log" > "$O/prod.tail.txt" 2>&1
tail -20 "$O/sab.log" > "$O/sab.tail.txt" 2>&1
