set -u
LANES=gp,gpc,gpc-multiclass
echo "== cpu column $(date)"
python3 tools/identity_break.py --lanes $LANES --repeats 2 --json "$LEG_OUT/cpu-x86.json" > "$LEG_OUT/cpu-x86.txt" 2>&1; echo "rc=$?"
echo "== host sabotage column $(date)"
MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1 python3 tools/identity_break.py --lanes $LANES --repeats 1 --json "$LEG_OUT/cpu-x86.host-sabotage.json" > "$LEG_OUT/cpu-x86.host-sabotage.txt" 2>&1; echo "rc=$?"
echo "== batch sabotage column $(date)"
MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1 python3 tools/identity_break.py --lanes $LANES --repeats 1 --json "$LEG_OUT/cpu-x86.batch-sabotage.json" > "$LEG_OUT/cpu-x86.batch-sabotage.txt" 2>&1; echo "rc=$?"
echo "== tests $(date)"
(cd python && python3 -m mojolearn.tests.test_gpc_surface) > "$LEG_OUT/test_gpc_surface.txt" 2>&1; echo "rc=$?"
.pixi/envs/test/bin/python -m pytest -q python/mojolearn/tests/test_host_surface.py python/mojolearn/tests/test_cpu_training_gp.py python/mojolearn/tests/test_cpu_inference_boundary.py > "$LEG_OUT/pytest.txt" 2>&1; echo "rc=$?"
tail -5 "$LEG_OUT"/*.txt
