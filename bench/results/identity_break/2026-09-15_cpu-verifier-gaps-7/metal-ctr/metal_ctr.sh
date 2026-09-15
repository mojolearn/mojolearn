#!/bin/bash
# lane/cpu-verifier-gaps-7: Metal-saved CTR table models for the six fixtures the
# committed models directory lacks (hashed, wide, denormal, denormal_ftz, dupes,
# negative). Uses the CTR lane's own HEAD Metal build (1386833b4), validated first
# on base against the committed model bytes and Metal hashes. One Metal job per
# chunk through the exclusive slot; one core.
set -u
SP=/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/e52730fc-0ee7-4bf3-8741-5fa0e0f1876f/scratchpad
WT=$SP/wt-cpu-gaps7
G=$SP/gaps7
PY=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python
L=$G/metal.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MOJOLEARN_CPU_THREADS=1 MODULAR_THREAD_BUSY_WAIT_US=0
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=$WT/python
cd $WT || exit 2
mkdir -p python/mojolearn/identical $G/models-validate $G/models $G/metal
cp $SP/ctr/bins/head/identical/_mojolearn.so python/mojolearn/identical/_mojolearn.so.tmp && mv python/mojolearn/identical/_mojolearn.so.tmp python/mojolearn/identical/_mojolearn.so
cp $SP/ctr/bins/head/identical/_mojolearn_gbdt.so python/mojolearn/identical/_mojolearn_gbdt.so.tmp && mv python/mojolearn/identical/_mojolearn_gbdt.so.tmp python/mojolearn/identical/_mojolearn_gbdt.so
shasum -a 256 python/mojolearn/identical/*.so >> $L
run() {  # lane fixtures repeats modeldir tag
  s=$(date +%s)
  MOJOLEARN_IDENTITY_GBDT_CTR_MODELS=$4 bash $SP/mac_slot.sh metal nice -n 19 $PY tools/identity_break.py \
    --lanes $1 --fixtures $2 --repeats $3 --vendor apple-m4 --json $G/metal/$5.json > $G/metal/$5.log 2>&1
  echo "$5 rc=$? $(( $(date +%s)-s ))s $(date +%T)" >> $L
}
EV=bench/results/identity_break/2026-09-15_gbdt-ctr-tables
echo "== validate $(date +%T)" >> $L
run gbdt-tensor-ctr-tables base 1 $G/models-validate validate-tensor
run gbdt-categorical-ctr-tables base 1 $G/models-validate validate-categorical
$PY - >> $L 2>&1 <<EOF
import json, hashlib, sys
ev = "$EV"; g = "$G"
ok = True
com = json.load(open(ev + "/apple-m4.json"))["cells"]
for lane, tag in (("gbdt-tensor-ctr-tables", "validate-tensor"), ("gbdt-categorical-ctr-tables", "validate-categorical")):
    a = hashlib.sha256(open(f"{ev}/models/{lane}.base.npz", "rb").read()).hexdigest()
    b = hashlib.sha256(open(f"{g}/models-validate/{lane}.base.npz", "rb").read()).hexdigest()
    new = json.load(open(f"{g}/metal/{tag}.json"))["cells"][f"{lane}/base"]
    old = com[f"{lane}/base"]
    same = a == b and all(new[k][0] == old[k][0] for k in ("infer", "model", "batch")) and new["hashes"][0] == old["hashes"][0]
    print(lane, "npz", a[:16], b[:16], "cells", "SAME" if same else "DIFFER", new["hashes"][0], old["hashes"][0])
    ok = ok and same
print("VALIDATE", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
[ $? -eq 0 ] || { echo "validation failed, stopping" >> $L; exit 1; }
echo "== new fixtures $(date +%T)" >> $L
NEWFX="hashed wide denormal denormal_ftz dupes negative"
run gbdt-tensor-ctr-tables hashed,wide,denormal,denormal_ftz,dupes,negative 2 $G/models tensor-6fx
for fx in $NEWFX; do
  run gbdt-categorical-ctr-tables $fx 1 $G/models categorical-$fx
done
ls -la $G/models >> $L
echo DONE >> $L
