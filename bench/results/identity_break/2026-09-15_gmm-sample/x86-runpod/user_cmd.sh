set -u
# Stage 2 CPU column on the RunPod CPU pod: GaussianMixture.sample.
# Runs in /root/mojolearn with PYTHONPATH=python, MOJOLEARN_NUMERIC_MODE=identical.
# --build mixture and --sabotage-build mixture have already placed
# python/mojolearn/host/ and python/mojolearn/host-sabotage/.
LANES=gmm,gmm-random-init,gmm-sample,gmm-random-init-sample
FIX=base,ties,odd,denormal
IB="python3 tools/identity_break.py --lanes $LANES --fixtures $FIX"
V=$(python3 -c 'import platform; print("cpu-x86-" + platform.machine())')
echo "== cpu $(date)"
$IB --repeats 2 --vendor "$V" --json "$LEG_OUT/cpu.json" > "$LEG_OUT/cpu.log" 2>&1; echo "cpu exit $?"
echo "== cpu_bsab $(date)"
MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1 $IB --repeats 1 --vendor "$V" --json "$LEG_OUT/cpu_bsab.json" > "$LEG_OUT/cpu_bsab.log" 2>&1; echo "cpu_bsab exit $?"
echo "== cpu_hsab $(date)"
MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
  $IB --repeats 1 --vendor "$V" --json "$LEG_OUT/cpu_hsab.json" > "$LEG_OUT/cpu_hsab.log" 2>&1; echo "cpu_hsab exit $?"
echo "== test_cpu $(date)"
(cd python && python3 -m mojolearn.tests.test_gmm_sample) > "$LEG_OUT/test_cpu.log" 2>&1; echo "test_cpu exit $?"
for f in cpu cpu_bsab cpu_hsab; do echo "---- $f"; grep -E "^cells=|^infer:|^batch:" "$LEG_OUT/$f.log"; done
tail -3 "$LEG_OUT/test_cpu.log"
