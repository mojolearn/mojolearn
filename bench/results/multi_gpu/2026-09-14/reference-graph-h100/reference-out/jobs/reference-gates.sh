#!/bin/bash
set -euo pipefail
export RUNPOD_POD_ID=4ra98lfm0pqum0 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
pixi run python tools/parallel_reference_neighbors_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/reference-out/report.json > /root/reference-out/public.log 2>&1
pixi run python tools/parallel_neighbors_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/reference-out/regression-report.json > /root/reference-out/regression.log 2>&1
pixi run python - <<'PY' > /root/reference-out/golden-comparison.log
import json
from pathlib import Path
old = json.loads(Path('/root/neighbors-out/report.json').read_text())
new = json.loads(Path('/root/reference-out/regression-report.json').read_text())
assert old == new
print('PASS: all 64 previous query receipts and output hashes are unchanged')
PY
