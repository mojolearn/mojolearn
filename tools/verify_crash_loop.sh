#!/bin/bash
# Repeat one verify cell in fresh processes and count worker crashes.
#
# usage: tools/verify_crash_loop.sh <tree|pypi:VERSION> <runs> <lane> <fixture[,fixture]>
#   tree          the branch tree's python/ (bindings built by the queue runner)
#   pypi:VERSION  a fresh venv with mojolearn[verify]==VERSION from PyPI (the published wheel)
# Prints one summary line: LOOP mode=.. lane=.. fixtures=.. runs=N ok=.. fail=.. crashed=.. exits=..
# Per-run logs go to ${LOOP_OUT:-$HOME/mq/out/verify-crash-loop}/<mode>/run-<i>.log.
set -u
mode=${1:?mode}; runs=${2:?runs}; lane=${3:?lane}; fixtures=${4:?fixtures}
out=${LOOP_OUT:-$HOME/mq/out/verify-crash-loop}/${mode//[^A-Za-z0-9.]/_}
mkdir -p "$out"
export MOJOLEARN_NUMERIC_MODE=identical
case $mode in
  tree)
    py="pixi run python"; export PYTHONPATH=$PWD/python ;;
  pypi:*)
    ver=${mode#pypi:}; venv=$out/venv
    if [ ! -x "$venv/bin/python" ]; then
      python3 -m venv "$venv" > "$out/venv.log" 2>&1 || { echo "LOOP mode=$mode venv-failed"; exit 3; }
      "$venv/bin/pip" install -q "mojolearn[verify]==$ver" >> "$out/venv.log" 2>&1 \
        || { echo "LOOP mode=$mode pip-failed (see $out/venv.log)"; exit 3; }
    fi
    py="$venv/bin/python"; unset PYTHONPATH ;;
  *) echo "unknown mode $mode"; exit 2 ;;
esac
ok=0; fail=0; crashed=0; exits=""
for i in $(seq 1 "$runs"); do
  log=$out/run-$i.log
  $py -m mojolearn verify --lanes "$lane" --fixtures "$fixtures" --json-out "$out/run-$i.json" > "$log" 2>&1
  rc=$?
  if grep -q -E 'worker exited|CRASHED' "$log"; then crashed=$((crashed + 1)); fi
  if [ $rc = 0 ]; then ok=$((ok + 1)); else fail=$((fail + 1)); exits="$exits $i:$rc"; fi
done
echo "LOOP mode=$mode lane=$lane fixtures=$fixtures runs=$runs ok=$ok fail=$fail crashed=$crashed exits=${exits:- none} logs=$out"
