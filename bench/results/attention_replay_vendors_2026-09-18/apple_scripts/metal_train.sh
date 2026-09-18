#!/bin/bash
# ONE Metal job (tools/mac_slot.sh metal): the reduced-shape HD64 training
# witness only (metal.sh's gates already ran; its binding path was wrong).
set -u
cd /Users/andrewhendel/mojolearn-wt/attention-replay-vendors
EV=/Users/andrewhendel/mojolearn-evidence/attention-replay-vendors/apple-native
O=$EV/out; V=$O/train_verdict.txt; : > "$V"
run() { name=$1; shift; t0=$(date +%s); "$@" > "$O/$name.log" 2>&1; c=$?; printf '%s\t%s\t%s\n' "$name" "$c" "$(( $(date +%s)-t0 ))" >> "$O/status.tsv"; return $c; }
# Reduced HD64 training witness (seed 20260917, enwik8, 100 steps).
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=$PWD/python
PY=$PWD/.pixi/envs/default/bin/python
CORP=/Users/andrewhendel/CascadeProjects/mojolearn/training/corpus/enwik8/input.txt
for arm in default bswz legacy; do
  so=$EV/../build-apple-$arm/_mojolearn_byte_lm.so
  [ -f "$so" ] || { echo "MISSING binding $arm" >> "$V"; continue; }
  cp "$so" python/mojolearn/identical/_mojolearn_byte_lm.so
  shasum -a 256 python/mojolearn/identical/_mojolearn_byte_lm.so >> "$O/binding_sha256.txt"
  run "probe3_$arm" $PY tools/lm_ce_alias_probe.py --out "$O/reduced3/$arm" --shape 1 256 128 2 2 64 256 2 256 \
      --steps 3 --tail 0 --smi-every 0 --witness-every 0 --corpus "$CORP"
  slow=$($PY -c "import json,sys;r=json.load(open('$O/reduced3/$arm/result.json'));print(int(max(x['seconds'] for x in r['steps'][1:])>10))" 2>>"$V")
  if [ "$slow" != 0 ]; then echo "STOP $arm: step over 10 s or probe failed (slow=$slow); not running 100 steps" >> "$V"; continue; fi
  run "train_$arm" $PY tools/lm_ce_alias_probe.py --out "$O/reduced/$arm" --shape 1 256 128 2 2 64 256 2 256 \
      --steps 100 --tail 0 --smi-every 0 --witness-every 25 --corpus "$CORP"
done
echo METAL-TRAIN-DONE >> "$O/status.tsv"
