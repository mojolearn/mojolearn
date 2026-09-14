#!/bin/bash
set -euo pipefail
export RUNPOD_POD_ID=rbtojh7e0esekh MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
pixi run python tools/parallel_boosting_adapters_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/pointwise-out/adapters-report.json > /root/pointwise-out/adapters.log 2>&1
pixi run python - <<'PY' > /root/pointwise-out/greedy-golden-comparison.log
import json
from pathlib import Path
p=Path('/root/pointwise-out')
a=json.loads((p/'previous-greedy-report.json').read_text())
b=json.loads((p/'greedy-regression-report.json').read_text())
assert a==b
print('PASS: all 16 prior greedy fixture receipts and output hashes unchanged')
PY
