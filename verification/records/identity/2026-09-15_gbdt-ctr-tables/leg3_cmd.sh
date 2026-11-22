set -u
# lane/inference-gbdt-ctr-tables, leg 3: the CPU column of the two CTR table
# lanes and its two negative controls, each sabotage binary the ONLY forest
# binary in its process (MOJOLEARN_HOST_DIR names a directory holding just
# it, so host_record cannot load the production one under the shared name).
V=$(python3 -c 'import platform; print("cpu-x86-" + platform.machine())')
EV=bench/results/identity_break/2026-09-15_gbdt-ctr-tables
R=bench/results/identity_break/2026-09-14_166-lanes
NEW=gbdt-categorical-ctr-tables,gbdt-tensor-ctr-tables
export MOJOLEARN_IDENTITY_GBDT_CTR_MODELS=$PWD/$EV/models
IB="python3 tools/identity_break.py --lanes $NEW --fixtures base,ties,odd"
COLS="$EV/apple-m4.json $R/nvidia-h100-sm_90a.json $R/amd-mi325x-gfx942.json"
echo "== ctr sabotage build $(date)"
env MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_GBDT_CTR_HOST_SABOTAGE=1" MOJOLEARN_FOREST_HOST_OUTDIR=python/mojolearn/host-ctr-sabotage \
  MOJOLEARN_BUILD_JOBS=8 sh bindings/build_forest_host.sh > "$LEG_OUT/build_ctr_sabotage.log" 2>&1; echo "ctr sabotage build exit $?"
ls python/mojolearn/host python/mojolearn/host-sabotage python/mojolearn/host-ctr-sabotage
sha256sum python/mojolearn/host*/_mojolearn_forest_host.so | tee "$LEG_OUT/forest_so_sha256.txt"
run() {  # name hostdir repeats
  MOJOLEARN_HOST_DIR=$PWD/$2 MOJOLEARN_FOREST_HOST_BINARY=$PWD/$2/_mojolearn_forest_host.so \
  MOJOLEARN_HOST_ALLOW_SABOTAGE=1 MOJOLEARN_FOREST_HOST_ALLOW_SABOTAGE=1 \
    $IB --repeats $3 --vendor "$V" --json "$LEG_OUT/$1.json" > "$LEG_OUT/$1.log" 2>&1
  echo "$1 exit $?"; grep -E "^cells=|^infer:|^batch:" "$LEG_OUT/$1.log"; grep -m1 "probe_error\|Error" "$LEG_OUT/$1.log" | cut -c1-200
}
echo "== columns $(date)"
run cpu python/mojolearn/host 2
run cpu-forest-sabotage python/mojolearn/host-sabotage 1
run cpu-ctr-sabotage python/mojolearn/host-ctr-sabotage 1
echo "== diffs $(date)"
python3 tools/identity_break.py --diff $COLS "$LEG_OUT/cpu.json" --require-columns 4 --lanes $NEW --owed-json "$LEG_OUT/owed.json" > "$LEG_OUT/diff-owed.txt" 2>&1
echo "diff production exit $?"; grep -E "^summary|^require-columns" "$LEG_OUT/diff-owed.txt"
for f in cpu-forest-sabotage cpu-ctr-sabotage; do
  python3 tools/identity_break.py --diff $COLS "$LEG_OUT/$f.json" --lanes $NEW > "$LEG_OUT/diff-$f.txt" 2>&1
  echo "diff $f exit $? (must be 1)"; grep -E "^summary" "$LEG_OUT/diff-$f.txt"
done
