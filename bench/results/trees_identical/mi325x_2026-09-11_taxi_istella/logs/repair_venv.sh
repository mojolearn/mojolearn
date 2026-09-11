#!/bin/sh
# Leg 2: the setup's apt step fetched stale package versions (404 on
# python3-venv, python3.12-venv, python3-pip-whl), so /root/venv-gpu was not
# created and amd_xgboost and its probe did not run (xgb_rocm_works=NO); batchC
# and batchD were stopped before any Istella-S cell finished. This refreshes
# apt and installs python3-venv, falls back to virtualenv from pip if apt
# still fails, installs amd_xgboost into a system-site venv, and re-runs the
# GPU probe beside a rocm-smi sampler, appending the verdict to setup.txt.
OUT=/root/trees_out
LOGS=$OUT/logs
VENV=/root/venv-gpu
export DEBIAN_FRONTEND=noninteractive
rm -rf $VENV
timeout -k 10 240 sh -c 'apt-get -o DPkg::Lock::Timeout=120 update && apt-get -o DPkg::Lock::Timeout=120 install -y --no-install-recommends python3-venv' > $LOGS/repair_apt.log 2>&1
echo "repair_apt=$? $(date -u +%H:%M:%S)" | tee -a $OUT/setup.txt
python3 -m venv --system-site-packages $VENV > $LOGS/repair_venv.log 2>&1
rc=$?
if [ "$rc" != 0 ]; then
    rm -rf $VENV
    python3 -m pip install --break-system-packages --no-input --disable-pip-version-check virtualenv >> $LOGS/repair_venv.log 2>&1
    python3 -m virtualenv --system-site-packages $VENV >> $LOGS/repair_venv.log 2>&1
    rc=$?
    echo "repair_venv_via=virtualenv" >> $OUT/setup.txt
fi
echo "repair_venv=$rc $(date -u +%H:%M:%S)" | tee -a $OUT/setup.txt
timeout -k 10 600 $VENV/bin/python -m pip install --no-input --disable-pip-version-check \
    --extra-index-url https://pypi.amd.com/rocm-6.4.4/simple amd_xgboost > $LOGS/repair_pip_amd_xgboost.log 2>&1
echo "repair_pip_amd_xgboost=$? $(date -u +%H:%M:%S)" | tee -a $OUT/setup.txt
( while :; do echo "t $(date -u +%T)"; rocm-smi --showuse --showmemuse 2>/dev/null | grep -i 'GPU use\|VRAM'; sleep 1; done ) > $LOGS/repair_xgb_probe_smi.log 2>&1 &
SMI=$!
timeout -k 10 300 $VENV/bin/python - > $LOGS/repair_xgb_probe.log 2>&1 <<'PY'
import json, time
import numpy as np
import xgboost as xgb
print("xgboost", xgb.__version__, xgb.__file__)
print("build_info", json.dumps({k: v for k, v in xgb.build_info().items() if k.startswith("USE")}))
rng = np.random.default_rng(0)
x = rng.normal(size=(1_000_000, 50)).astype(np.float32)
y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(np.float32)
for dev in ("cuda", "cpu", "cuda"):
    t = time.perf_counter()
    m = xgb.XGBClassifier(n_estimators=100, max_depth=6, tree_method="hist", device=dev, verbosity=1)
    m.fit(x, y)
    ms = (time.perf_counter() - t) * 1e3
    try:
        cfg = json.loads(m.get_booster().save_config())["learner"]["generic_param"]["device"]
    except Exception as exc:  # noqa: BLE001
        cfg = "unreadable(%s)" % exc.__class__.__name__
    print("fit device=%s %.1f ms config_device=%s" % (dev, ms, cfg))
print("XGB_GPU_PROBE_DONE")
PY
kill $SMI 2>/dev/null
if grep -q XGB_GPU_PROBE_DONE $LOGS/repair_xgb_probe.log \
   && ! grep -qi 'not compiled\|falling back\|fall back\|changed from GPU\|no visible GPU' $LOGS/repair_xgb_probe.log \
   && awk -F: '/GPU use/ { v = $NF; gsub(/[^0-9]/, "", v); if (v + 0 > 0) f = 1 } END { exit !f }' $LOGS/repair_xgb_probe_smi.log; then
    echo "xgb_rocm_works=yes (repair $(date -u +%H:%M:%S))" | tee -a $OUT/setup.txt
else
    echo "xgb_rocm_works_repair=NO $(date -u +%H:%M:%S)" | tee -a $OUT/setup.txt
fi
$VENV/bin/python -c "import xgboost, lightgbm, catboost, sklearn; print('repair venv xgboost', xgboost.__version__, xgboost.__file__, 'lightgbm', lightgbm.__version__, lightgbm.__file__)" >> $OUT/versions.txt 2>&1
tail -1 $OUT/versions.txt
