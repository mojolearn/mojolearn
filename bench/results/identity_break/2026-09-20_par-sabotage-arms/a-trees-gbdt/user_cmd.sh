#!/bin/bash
# lane/gpu-confirm-never-launched, shard a (trees / gbdt / forest par drivers).
#
# ONE clean CPU column and ONE sabotage-BUILD column over the same par-* lanes,
# BOTH at --repeats 2, so the negative control meets
# tools/verification_matrix.py's stable-digest rule (a one-repeat sabotage
# column has its move SILENTLY DISCARDED, which is how 135 lanes' worth of
# real negative controls were lost once already).
#
# These lanes stood at sabotage="none" or "declared": the switch exists (or
# the arithmetic is reachable) but NOBODY HAS WATCHED IT FAIL. This is the run
# that watches. `seen(build)` is decided empirically by
# verification_matrix.sabotage_moves -- a sabotage column paired against a
# clean column of the same device class whose HASHES DIFFER -- so a lane that
# is in no family's lane list still earns the credit if its bytes move.
set -u
O="$LEG_OUT"
LANES='par-boosting,par-boosting-clf,par-boosting-pointwise,par-boosting-reg,par-border-types,par-cross-val,par-feature-freq,par-forest-et-clf,par-forest-pool,par-forest-reg,par-iforest,par-ordered,par-ordered-rmse'
FIX=base,ties
st() { n=$1; shift; "$@"; rc=$?; printf '%s\t%s\n' "$n" "$rc" >> "$O/step_exit_codes.txt"; return 0; }
echo "shard a nproc=$(nproc) $(uname -m) $(python3 --version 2>&1)" > "$O/box.txt"
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
python3 - <<'PYEOF' > "$O/arm_bytes_differ.txt" 2>&1
import hashlib, os
p = "python/mojolearn/host"; q = "python/mojolearn/host-sabotage"
def h(f): return hashlib.sha256(open(f, 'rb').read()).hexdigest()
same = []
for b in sorted(os.listdir(q)):
    if not b.endswith(".so"):
        continue
    a = os.path.join(p, b)
    if not os.path.exists(a):
        continue
    eq = h(a) == h(os.path.join(q, b))
    if eq:
        same.append(b)
    print(("SAME  " if eq else "DIFFER"), b)
print("identical-byte arms:", len(same), same)
PYEOF
SAB="MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1"
SAB="$SAB MOJOLEARN_FOREST_HOST_BINARY=python/mojolearn/host-sabotage/_mojolearn_forest_host.so MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE=1"
SAB="$SAB MOJOLEARN_BYTE_LM_HOST_BINARY=python/mojolearn/host-sabotage/_mojolearn_byte_lm_host.so MOJOLEARN_BYTE_LM_HOST_ALLOW_SABOTAGE=1"
st prod python3 tools/identity_break.py --lanes "$LANES" --fixtures $FIX --repeats 2 --json "$O/cpu-x86.json" > "$O/prod.log" 2>&1
st sab  env $SAB python3 tools/identity_break.py --lanes "$LANES" --fixtures $FIX --repeats 2 --json "$O/cpu-x86.sabotage.json" > "$O/sab.log" 2>&1
st diff python3 tools/identity_break.py --diff "$O/cpu-x86.json" "$O/cpu-x86.sabotage.json" > "$O/diff.clean-vs-sabotage.txt" 2>&1
tail -30 "$O/prod.log" > "$O/prod.tail.txt" 2>&1
tail -30 "$O/sab.log" > "$O/sab.tail.txt" 2>&1
