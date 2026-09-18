#!/bin/bash
# lane/sabotage-sweep, shard b-trees-gbdt: one clean CPU column and one sabotage-BUILD
# column over the same lanes, BOTH at --repeats 2, so the negative control
# meets tools/verification_matrix.py's stable-digest rule (a one-repeat
# sabotage column is what every earlier record has, and the matrix discards
# every one of them).
set -u
O="$LEG_OUT"
LANES='cross-val,et-clf-entropy-bestfirst,et-reg-bootstrap-parallel,gbdt-adapter-clf,gbdt-adapter-reg,gbdt-adapter-score-weighted,gbdt-categorical-ctr,gbdt-categorical-ctr-tables,gbdt-depthwise,gbdt-exact-mae,gbdt-feature-freq,gbdt-lossguide,gbdt-lossguide-newtoncosine,gbdt-multiclass,gbdt-nan-modes,gbdt-onevsall,gbdt-ordered-rmse,gbdt-pair-logit,gbdt-parametric-losses,gbdt-pointwise-l2-bayesian-eval,gbdt-query-rmse,gbdt-rmse,gbdt-symmetric,gbdt-tensor-ctr-tables,gbdt-yeti-rank,iforest-tuned,par-forest,par-forest-et,rf-clf,rf-clf-balanced-parallel,rf-clf-entropy-log2-noboot,rf-reg,rf-reg-gamma-ig,rf-reg-poisson,rf-score-weighted'
FIX=base,ties
st() { n=$1; shift; "$@"; rc=$?; printf '%s\t%s\n' "$n" "$rc" >> "$O/step_exit_codes.txt"; return 0; }
echo "shard b-trees-gbdt nproc=$(nproc) $(uname -m) $(python3 --version 2>&1)" > "$O/box.txt"
echo "$LANES" | tr ',' '\n' > "$O/lanes.txt"
sha256sum python/mojolearn/host/*.so python/mojolearn/host-sabotage/*.so > "$O/so_sha256.txt" 2>&1
# Both sets must hold every declared binding. A missing one loads nothing and
# reads REFUSED, which a diff would miscount as a catch.
: > "$O/missing_bindings.txt"
for b in $(python3 python/mojolearn/host_surface.py --bindings --sep ' '); do
  [ -f "python/mojolearn/host/$b.so" ] || echo "PROD $b" >> "$O/missing_bindings.txt"
  [ -f "python/mojolearn/host-sabotage/$b.so" ] || echo "SAB $b" >> "$O/missing_bindings.txt"
done
# The two arms must be DIFFERENT binaries. Same bytes = the define did nothing.
python3 - "$O" <<'PYEOF' > "$O/arm_bytes_differ.txt" 2>&1
import hashlib, os, sys
p="python/mojolearn/host"; q="python/mojolearn/host-sabotage"
def h(f): return hashlib.sha256(open(f,'rb').read()).hexdigest()
same=[]
for b in sorted(os.listdir(q)):
    if not b.endswith(".so"): continue
    a=os.path.join(p,b)
    if not os.path.exists(a): continue
    if h(a)==h(os.path.join(q,b)): same.append(b)
    print(("SAME  " if h(a)==h(os.path.join(q,b)) else "DIFFER"), b)
print("identical-byte arms:", len(same), same)
PYEOF
SAB="MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1"
SAB="$SAB MOJOLEARN_FOREST_HOST_BINARY=python/mojolearn/host-sabotage/_mojolearn_forest_host.so MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE=1"
SAB="$SAB MOJOLEARN_BYTE_LM_HOST_BINARY=python/mojolearn/host-sabotage/_mojolearn_byte_lm_host.so MOJOLEARN_BYTE_LM_HOST_ALLOW_SABOTAGE=1"
st prod python3 tools/identity_break.py --lanes "$LANES" --fixtures $FIX --repeats 2 --json "$O/cpu-x86.json" > "$O/prod.log" 2>&1
st sab  env $SAB python3 tools/identity_break.py --lanes "$LANES" --fixtures $FIX --repeats 2 --json "$O/cpu-x86.sabotage.json" > "$O/sab.log" 2>&1
st diff python3 tools/identity_break.py --diff "$O/cpu-x86.json" "$O/cpu-x86.sabotage.json" > "$O/diff.clean-vs-sabotage.txt" 2>&1
tail -25 "$O/prod.log" > "$O/prod.tail.txt" 2>&1
tail -25 "$O/sab.log" > "$O/sab.tail.txt" 2>&1
