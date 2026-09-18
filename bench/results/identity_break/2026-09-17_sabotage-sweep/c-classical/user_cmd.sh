#!/bin/bash
# lane/sabotage-sweep, shard c-classical: one clean CPU column and one sabotage-BUILD
# column over the same lanes, BOTH at --repeats 2, so the negative control
# meets tools/verification_matrix.py's stable-digest rule (a one-repeat
# sabotage column is what every earlier record has, and the matrix discards
# every one of them).
set -u
O="$LEG_OUT"
LANES='arima,arima-011,arima-seasonal-c,bootstrap,gemm-bf16,gemm-int8,gemm-pinned,gemm-transposed,gmm-random-init-sample,gmm-sample,gp-normalize-y,gp-optimize,gp-optimize-restarts,gp-sample-y,gp-sample-y-normalize,gpc,gpc-multiclass,metrics,metrics-classification,monte-carlo,par-arima,par-holtwinters,permutation-test,spectral-precomputed,umap'
FIX=base,ties
st() { n=$1; shift; "$@"; rc=$?; printf '%s\t%s\n' "$n" "$rc" >> "$O/step_exit_codes.txt"; return 0; }
echo "shard c-classical nproc=$(nproc) $(uname -m) $(python3 --version 2>&1)" > "$O/box.txt"
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
