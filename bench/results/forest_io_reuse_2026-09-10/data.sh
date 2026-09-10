#!/bin/bash
set -euo pipefail
exec > /root/forest_out/data-setup.log 2>&1
trap 'echo "data_setup_exit=$?" > /root/forest_out/data-setup.exit' EXIT
cd /root/mojolearn
export GBM_BENCH_DATA=/root/datasets/gbm-bench
python3 tools/speed_gbdt_arm.py --download year
python3 - <<'PY'
import sys
sys.path.insert(0,'tools')
import speed_gbdt_arm as s
for name in ('year','covtype'):
 d=s.load_dataset(name,'shipped',500000 if name=='year' else None)
 print(name,d.X_train.shape,d.X_test.shape,flush=True)
PY
touch /root/forest_out/data-setup.done
