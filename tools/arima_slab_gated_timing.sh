#!/bin/bash
# ARIMA_SLAB timing, only after tools/arima_slab_quality.sh PASS;
# manager verified_arms.py validates/stages the M2 arms anew.
# Usage: bash tools/arima_slab_gated_timing.sh SOURCE QUALITY_TAG TIMING_TAG DATASET
set -euo pipefail
arima_source=${1:?source SHA}; arima_quality=${2:?quality tag}
arima_timing=${3:?timing tag}; arima_dataset=${4:?dataset}
[[ "$arima_quality" =~ ^[A-Za-z0-9_.-]+$ && "$arima_timing" =~ ^[A-Za-z0-9_.-]+$ ]]
[[ "$arima_dataset" = synthetic || "$arima_dataset" = taxi-hourly ]]
cd "$(dirname "$0")/.."
arima_python=$HOME/board-0834/cache/venv/bin/python
"$arima_python" - "$arima_source" "$arima_quality" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
source, tag = sys.argv[1:]
receipt = json.loads((Path.home() / 'mq/out' / ('arima-quality-' + tag) / 'FULL_PASS.json').read_text())
assert receipt['status'] == 'PASS' and receipt['source_sha'] == source
assert receipt['fixtures'] == ['seed23_n512', 'seed87_n2048', 'seed61_n1392']
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == source
root = Path.home() / 'mq/verified-arms' / source / 'arima'
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['hashes'] == receipt['hashes']
for arm in ('A', 'B'):
    assert hashlib.sha256((root / (arm + '.so')).read_bytes()).hexdigest() == receipt['hashes'][arm]
PY
# Missing/failed/stale gate exits above; this helper cannot build or rerun a
# nonempty scored race (verified_arms.py enforces that refusal).
exec "$arima_python" "$HOME/mq/verified_arms.py" "$arima_source" arima MOJOLEARN_ARIMA_SLAB "$arima_timing" \
  bash tools/afc_ab_def.sh "$arima_timing" arima autoarima "$arima_dataset" 1 1 '' '-D MOJOLEARN_ARIMA_SLAB'
