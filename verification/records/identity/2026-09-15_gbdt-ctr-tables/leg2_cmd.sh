set -u
# lane/inference-gbdt-ctr-tables, leg 2: the CPU columns of the two CTR table
# lanes from the Metal-saved models, the forest and CTR sabotage columns, the
# tests, and the installed test wheel. Runs in /root/mojolearn with
# PYTHONPATH=python, MOJOLEARN_NUMERIC_MODE=identical; --build forest and
# --sabotage-build forest (-D MOJOLEARN_FOREST_HOST_SABOTAGE=1) have placed
# python/mojolearn/host/ and python/mojolearn/host-sabotage/.
V=$(python3 -c 'import platform; print("cpu-x86-" + platform.machine())')
EV=bench/results/identity_break/2026-09-15_gbdt-ctr-tables
NEW=gbdt-categorical-ctr-tables,gbdt-tensor-ctr-tables
FIX=base,ties,odd
export MOJOLEARN_IDENTITY_GBDT_CTR_MODELS=$PWD/$EV/models
IB="python3 tools/identity_break.py --lanes $NEW --fixtures $FIX"
PROD=$PWD/python/mojolearn/host/_mojolearn_forest_host.so
FSAB=$PWD/python/mojolearn/host-sabotage/_mojolearn_forest_host.so
CSAB=$PWD/python/mojolearn/host-ctr-sabotage/_mojolearn_forest_host.so

echo "== ctr sabotage build $(date)"
env MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_GBDT_CTR_HOST_SABOTAGE=1" MOJOLEARN_FOREST_HOST_OUTDIR=python/mojolearn/host-ctr-sabotage \
  MOJOLEARN_BUILD_JOBS=8 sh bindings/build_forest_host.sh > "$LEG_OUT/build_ctr_sabotage.log" 2>&1; echo "ctr sabotage build exit $?"
sha256sum python/mojolearn/host*/_mojolearn_forest_host.so | tee "$LEG_OUT/forest_so_sha256.txt"

# the host families built here and no GPU set: the package takes its CPU route;
# only the forest binding is ever named, so no other host binding can predict
echo "== cpu column (production forest) $(date)"
MOJOLEARN_FOREST_HOST_BINARY=$PROD $IB --repeats 2 --vendor "$V" --json "$LEG_OUT/cpu.json" > "$LEG_OUT/cpu.log" 2>&1; echo "cpu exit $?"
echo "== forest sabotage column $(date)"
MOJOLEARN_FOREST_HOST_BINARY=$FSAB MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE=1 $IB --repeats 1 --vendor "$V" --json "$LEG_OUT/cpu-forest-sabotage.json" > "$LEG_OUT/cpu-forest-sabotage.log" 2>&1; echo "forest sabotage exit $?"
echo "== ctr sabotage column $(date)"
MOJOLEARN_FOREST_HOST_BINARY=$CSAB MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE=1 $IB --repeats 1 --vendor "$V" --json "$LEG_OUT/cpu-ctr-sabotage.json" > "$LEG_OUT/cpu-ctr-sabotage.log" 2>&1; echo "ctr sabotage exit $?"
for f in cpu cpu-forest-sabotage cpu-ctr-sabotage; do echo "---- $f"; grep -E "^cells=|^infer:|^batch:" "$LEG_OUT/$f.log"; done

echo "== diffs $(date)"
python3 tools/identity_break.py --diff $EV/apple-m4.json "$LEG_OUT/cpu.json" --require-columns 4 --lanes $NEW \
  --owed-json "$LEG_OUT/owed.json" > "$LEG_OUT/diff-metal-cpu.txt" 2>&1; echo "diff metal vs cpu (require 4, owed) exit $?"
grep -E "^summary|^require-columns" "$LEG_OUT/diff-metal-cpu.txt"
for f in cpu-forest-sabotage cpu-ctr-sabotage; do
  python3 tools/identity_break.py --diff $EV/apple-m4.json "$LEG_OUT/$f.json" --lanes $NEW > "$LEG_OUT/diff-metal-$f.txt" 2>&1
  echo "diff metal vs $f exit $? (must be 1)"; grep -E "^summary" "$LEG_OUT/diff-metal-$f.txt"
done

echo "== tests $(date)"
(cd python && MOJOLEARN_FOREST_HOST_BINARY=$PROD python3 -m mojolearn.tests.test_gbdt_host_ctr) > "$LEG_OUT/test_gbdt_host_ctr.log" 2>&1; echo "test_gbdt_host_ctr exit $?"; tail -2 "$LEG_OUT/test_gbdt_host_ctr.log"
(cd python && MOJOLEARN_FOREST_HOST_BINARY=$CSAB python3 -m mojolearn.tests.test_gbdt_host_ctr) > "$LEG_OUT/test_gbdt_host_ctr.ctr-sabotage-refused.log" 2>&1; echo "ctr sabotage without the switch exit $? (must be 1, refused)"; grep -m1 "SABOTAGE" "$LEG_OUT/test_gbdt_host_ctr.ctr-sabotage-refused.log"
(cd python && MOJOLEARN_FOREST_HOST_BINARY=$CSAB MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE=1 python3 -m mojolearn.tests.test_gbdt_host_ctr) > "$LEG_OUT/test_gbdt_host_ctr.ctr-sabotage.log" 2>&1; echo "test under ctr sabotage exit $? (must be 1)"; grep -m1 "AssertionError" "$LEG_OUT/test_gbdt_host_ctr.ctr-sabotage.log"
(cd python && MOJOLEARN_FOREST_HOST_BINARY=$PROD python3 -m mojolearn.tests.test_gbdt_host_modes) > "$LEG_OUT/test_gbdt_host_modes.log" 2>&1; echo "test_gbdt_host_modes exit $?"; tail -1 "$LEG_OUT/test_gbdt_host_modes.log"

echo "== installed test wheel $(date)"
T=/root/wheel-target; rm -rf $T python/dist python/build; mkdir -p $T
cp LICENSE NOTICE README.md python/ 2>/dev/null; cp CITATION.cff python/mojolearn/ 2>/dev/null
rm -rf python/mojolearn/host-sabotage.aside; mv python/mojolearn/host-sabotage python/mojolearn/host-sabotage.aside
mv python/mojolearn/host-ctr-sabotage python/mojolearn/host-ctr-sabotage.aside
# the package-data glob ships every .so on disk: only the forest host binding
# goes into this test wheel, as in lane/inference-gbdt-modes
mkdir -p /root/host-aside
for so in python/mojolearn/host/*.so; do
  case "$so" in */_mojolearn_forest_host.so) ;; *) mv "$so" /root/host-aside/ ;; esac
done
ls python/mojolearn/host/ | tee "$LEG_OUT/wheel-host-dir.txt"
(cd python && ../.pixi/envs/pkg/bin/python -m build --wheel --no-isolation) > "$LEG_OUT/wheel_build.log" 2>&1; echo "wheel build exit $?"
W=$(ls python/dist/*.whl | head -1); echo "wheel $W"
python3 -m zipfile -l "$W" | grep -E "host/|_gbdt_host|tensor" | tee "$LEG_OUT/wheel-contents.txt"
python3 -m zipfile -e "$W" $T
(cd /root && PYTHONPATH=$T WT=/root/mojolearn python3 /root/mojolearn/tools/inf_gbdt_wheel_models.py check-saved /root/mojolearn/$EV/models /root/mojolearn/$EV/apple-m4.json) > "$LEG_OUT/wheel-check.txt" 2>&1; echo "wheel check exit $?"; tail -2 "$LEG_OUT/wheel-check.txt"; grep -m2 "mojolearn from\|forest binary" "$LEG_OUT/wheel-check.txt"
(cd /root && PYTHONPATH=$T WT=/root/mojolearn MOJOLEARN_FOREST_HOST_BINARY=$PWD/mojolearn/python/mojolearn/host-ctr-sabotage.aside/_mojolearn_forest_host.so MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE=1 python3 /root/mojolearn/tools/inf_gbdt_wheel_models.py check-saved /root/mojolearn/$EV/models /root/mojolearn/$EV/apple-m4.json) > "$LEG_OUT/wheel-check-ctr-sabotage.txt" 2>&1; echo "wheel check under ctr sabotage exit $? (must be 1)"; tail -1 "$LEG_OUT/wheel-check-ctr-sabotage.txt"
