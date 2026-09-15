set -u
# CPU column, host sabotage column and tests for GaussianProcessRegressor.sample_y
# (lane/gp-sample-y). Runs in /root/mojolearn with PYTHONPATH=python and
# MOJOLEARN_NUMERIC_MODE=identical; --build gp,preprocessing and
# --sabotage-build gp,preprocessing have placed host/ and host-sabotage/.
LANES=gp,gp-matern12,gp-matern32,gp-matern52-ard,gp-normalize-y,gp-sample-y,gp-sample-y-normalize
IB="python3 tools/identity_break.py --lanes $LANES"
V=$(python3 -c 'import platform; print("cpu-x86-" + platform.machine())')
echo "== cpu $(date)"
$IB --repeats 2 --vendor "$V" --json "$LEG_OUT/cpu.json" > "$LEG_OUT/cpu.log" 2>&1; echo "cpu exit $?"
echo "== cpu_hsab $(date)"
MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
  $IB --repeats 1 --vendor "$V" --json "$LEG_OUT/cpu_hsab.json" > "$LEG_OUT/cpu_hsab.log" 2>&1; echo "cpu_hsab exit $?"
echo "== tests $(date)"
(cd python && python3 -m mojolearn.tests.test_gp_sample_y) > "$LEG_OUT/test_gp_sample_y.log" 2>&1; echo "test_gp_sample_y exit $?"
(cd python && python3 -m mojolearn.tests.test_cpu_training_gp) > "$LEG_OUT/test_cpu_training_gp.log" 2>&1; echo "test_cpu_training_gp exit $?"
(cd python && ../.pixi/envs/test/bin/python -m pytest -q mojolearn/tests/test_gp_normalize_y.py) > "$LEG_OUT/test_gp_normalize_y.log" 2>&1; echo "test_gp_normalize_y exit $?"
for f in cpu cpu_hsab; do echo "---- $f"; grep -E "^cells=|^infer:|^batch:" "$LEG_OUT/$f.log"; done
for f in test_gp_sample_y test_cpu_training_gp test_gp_normalize_y; do echo "---- $f"; tail -n 3 "$LEG_OUT/$f.log"; done
true
