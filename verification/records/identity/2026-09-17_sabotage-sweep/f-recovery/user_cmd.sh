#!/bin/bash
# lane/sabotage-sweep, shard f: the lanes shards b, c and d could not answer.
# Five refused for a host family that shard's build did not carry (metrics for
# cross-val, gbdt-adapter-score-weighted and rf-score-weighted; preprocessing
# for the two gp normalize lanes). Two refused because their CTR fixtures live
# under bench/results, which the leg leaves home unless --include names it.
# And ordered-gradient-sum read INERT on base and ties in shard d, so it is
# asked again on ALL NINE fixtures: an arm that fires nowhere is a finding,
# and an arm that fires on a fixture two of them do not reach is a different
# one.
set -u
O="$LEG_OUT"
LANES='cross-val,gbdt-adapter-score-weighted,rf-score-weighted,gp-normalize-y,gp-sample-y-normalize,gbdt-categorical-ctr-tables,gbdt-tensor-ctr-tables'
st() { n=$1; shift; "$@"; rc=$?; printf '%s\t%s\n' "$n" "$rc" >> "$O/step_exit_codes.txt"; return 0; }
mkdir -p "$O/f-recovery" "$O/g-ordered-gradient-sum"
echo "shard f nproc=$(nproc) $(uname -m) $(python3 --version 2>&1)" > "$O/f-recovery/box.txt"
cp "$O/f-recovery/box.txt" "$O/g-ordered-gradient-sum/box.txt"
echo "$LANES" | tr ',' '\n' > "$O/f-recovery/lanes.txt"
echo ordered-gradient-sum > "$O/g-ordered-gradient-sum/lanes.txt"
sha256sum python/mojolearn/host/*.so python/mojolearn/host-sabotage/*.so > "$O/f-recovery/so_sha256.txt" 2>&1
cp "$O/f-recovery/so_sha256.txt" "$O/g-ordered-gradient-sum/so_sha256.txt"
python3 - <<'PYEOF' > "$O/f-recovery/arm_bytes_differ.txt" 2>&1
import hashlib, os
p="python/mojolearn/host"; q="python/mojolearn/host-sabotage"
def h(f): return hashlib.sha256(open(f,'rb').read()).hexdigest()
same=[]
for b in sorted(os.listdir(q)):
    if not b.endswith(".so"): continue
    a=os.path.join(p,b)
    if not os.path.exists(a): continue
    eq = h(a)==h(os.path.join(q,b))
    same += [b] if eq else []
    print(("SAME  " if eq else "DIFFER"), b)
print("identical-byte arms:", len(same), same)
PYEOF
cp "$O/f-recovery/arm_bytes_differ.txt" "$O/g-ordered-gradient-sum/arm_bytes_differ.txt"
ls bench/results/identity_break/2026-09-15_gbdt-ctr-tables/models/ > "$O/f-recovery/ctr_models_present.txt" 2>&1
SAB="MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1"
SAB="$SAB MOJOLEARN_FOREST_HOST_BINARY=python/mojolearn/host-sabotage/_mojolearn_forest_host.so MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE=1"
F="$O/f-recovery"; G="$O/g-ordered-gradient-sum"
st f_prod python3 tools/identity_break.py --lanes "$LANES" --fixtures base,ties --repeats 2 --json "$F/cpu-x86.json" > "$F/prod.log" 2>&1
st f_sab  env $SAB python3 tools/identity_break.py --lanes "$LANES" --fixtures base,ties --repeats 2 --json "$F/cpu-x86.sabotage.json" > "$F/sab.log" 2>&1
st f_diff python3 tools/identity_break.py --diff "$F/cpu-x86.json" "$F/cpu-x86.sabotage.json" > "$F/diff.clean-vs-sabotage.txt" 2>&1
st g_prod python3 tools/identity_break.py --lanes ordered-gradient-sum --repeats 2 --json "$G/cpu-x86.json" > "$G/prod.log" 2>&1
st g_sab  env $SAB python3 tools/identity_break.py --lanes ordered-gradient-sum --repeats 2 --json "$G/cpu-x86.sabotage.json" > "$G/sab.log" 2>&1
st g_diff python3 tools/identity_break.py --diff "$G/cpu-x86.json" "$G/cpu-x86.sabotage.json" > "$G/diff.clean-vs-sabotage.txt" 2>&1
for d in "$F" "$G"; do tail -30 "$d/prod.log" > "$d/prod.tail.txt" 2>&1; tail -30 "$d/sab.log" > "$d/sab.tail.txt" 2>&1; done
# The bpe-trainer lane's docstring says its tie-break arm leaves n_tokens and
# n_merges alone. Shard e moved all five parts, so read the COUNTS themselves.
st bpe_counts python3 - <<'PYEOF' > "$O/bpe_counts.txt" 2>&1
import importlib, importlib.util, os, sys, numpy as np
sys.path.insert(0, "python")
spec = importlib.util.spec_from_file_location("ib", "tools/identity_break.py")
ib = importlib.util.module_from_spec(spec); sys.modules["ib"] = ib; spec.loader.exec_module(ib)
import mojolearn as ml
from mojolearn import _bpe_trainer
for fx in ("base", "ties"):
    X = ib.fixture(fx)[0]
    raw = np.ascontiguousarray(X).tobytes()[:4096]
    for tag, val in (("clean", None), ("sabotage", "1")):
        if val: os.environ["MOJOLEARN_BPE_TRAINER_SABOTAGE"] = val
        else: os.environ.pop("MOJOLEARN_BPE_TRAINER_SABOTAGE", None)
        importlib.reload(_bpe_trainer)
        v = ml.tokenizer.BpeVocabularyTrainer(vocab_size=320, min_frequency=2).train([raw])
        print(f"{fx} {tag}: sabotaged={_bpe_trainer.sabotaged()} n_tokens={v.n_tokens} "
              f"n_merges={len(v.merges)} n_ties_broken={v.n_ties_broken}")
PYEOF
