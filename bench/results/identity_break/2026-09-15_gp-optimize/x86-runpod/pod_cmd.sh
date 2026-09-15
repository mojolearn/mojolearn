set -u
# CPU column, gradient-sabotage column and tests for the gp optimizer lane.
# Runs in /root/mojolearn with PYTHONPATH=python and MOJOLEARN_NUMERIC_MODE=identical;
# --build gp,preprocessing,core placed python/mojolearn/host, and
# --sabotage-build gp with -D MOJOLEARN_GP_GRAD_SABOTAGE=1 placed host-sabotage.
LANES=gp,gp-matern12,gp-matern32,gp-matern52-ard,gp-normalize-y,gp-optimize,gp-optimize-restarts
IB="python3 tools/identity_break.py --lanes $LANES"
V=$(python3 -c 'import platform; print("cpu-x86-" + platform.machine())')
echo "== cpu $(date)"
$IB --repeats 2 --vendor "$V" --json "$LEG_OUT/cpu.json" > "$LEG_OUT/cpu.log" 2>&1; echo "cpu exit $?"
echo "== cpu_gsab $(date)"
MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
  $IB --repeats 1 --vendor "$V" --json "$LEG_OUT/cpu_gsab.json" > "$LEG_OUT/cpu_gsab.log" 2>&1; echo "cpu_gsab exit $?"
echo "== tests $(date)"
(cd python && python3 -m mojolearn.tests.test_gp_optimizer) > "$LEG_OUT/test_gp_optimizer.log" 2>&1; echo "test_gp_optimizer exit $?"
(cd python && python3 -m mojolearn.tests.test_gp_surface) > "$LEG_OUT/test_gp_surface.log" 2>&1; echo "test_gp_surface exit $?"
(cd python && python3 -m mojolearn.tests.test_gp_sample_y) > "$LEG_OUT/test_gp_sample_y.log" 2>&1; echo "test_gp_sample_y exit $?"
(cd python && ../.pixi/envs/test/bin/python -m pytest -q mojolearn/tests/test_host_surface.py) > "$LEG_OUT/test_host_surface.log" 2>&1; echo "test_host_surface exit $?"
for f in cpu cpu_gsab; do echo "---- $f"; grep -E "^cells=|^infer:|^batch:" "$LEG_OUT/$f.log"; done
for f in test_gp_optimizer test_gp_surface test_gp_sample_y test_host_surface; do echo "---- $f"; tail -n 3 "$LEG_OUT/$f.log"; done
true
