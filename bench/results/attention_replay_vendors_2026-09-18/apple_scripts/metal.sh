#!/bin/bash
# ONE Metal job (run under tools/mac_slot.sh metal): native gates, then the
# reduced-shape HD64 training witness. Never run outside the Metal slot.
set -u
cd /Users/andrewhendel/mojolearn-wt/attention-replay-vendors
EV=/Users/andrewhendel/mojolearn-evidence/attention-replay-vendors/apple-native
B=$EV/bin; O=$EV/out; mkdir -p "$O"
V=$O/gates_verdict.txt; : > "$V"
run() { name=$1; shift; t0=$(date +%s); "$@" > "$O/$name.log" 2>&1; c=$?; printf '%s\t%s\t%s\n' "$name" "$c" "$(( $(date +%s)-t0 ))" >> "$O/status.tsv"; return $c; }
must_fail() { name=$1; pat=$2
  if run "$name" "$B/$name"; then echo "BLIND: $name passed" >> "$V"; return; fi
  if grep "$pat" "$O/$name.log" >> "$V"; then echo "EXPECTED FAIL $name" >> "$V"; else echo "WRONG FAILURE $name (no '$pat')" >> "$V"; fi; }
must_fail masked_rowoff 'masked-tail repair disabled: INERT'
must_fail masked_sab_z 'zdot repair differs'
must_fail masked_sab_dq 'dq repair differs'
must_fail masked_corrupt_z 'zdot preserve differs'
must_fail masked_corrupt_dq 'dq preserve differs'
must_fail tail_sab 'accepted dk differs'
for g in masked_clean tail_clean fused_default; do
  if run "$g" "$B/$g"; then echo "PASS $g" >> "$V"; else echo "FAIL $g" >> "$V"; fi
done
# Reduced HD64 training witness (seed 20260917, enwik8, 100 steps).
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=$PWD/python
PY=$PWD/.pixi/envs/default/bin/python
CORP=/Users/andrewhendel/CascadeProjects/mojolearn/training/corpus/enwik8/input.txt
for arm in default bswz legacy; do
  so=$EV/build-apple-$arm/_mojolearn_byte_lm.so
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
echo METAL-DONE >> "$O/status.tsv"
