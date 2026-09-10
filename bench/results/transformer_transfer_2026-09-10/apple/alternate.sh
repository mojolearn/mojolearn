#!/bin/bash
set -euo pipefail
cd /Users/andrewhendel/CascadeProjects/mojolearn
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python
out=/tmp/mojolearn-transformer-transfer-apple
for item in caller.2 baseline.2 baseline.3 caller.3; do
 arm=${item%.*}
 cp "$out/$arm.so" python/mojolearn/identical/_mojolearn_transformer.so
 python tools/transformer_transfer_check.py --output "$out/$item.json" > "$out/$item.log" 2>&1
done
python - <<'PY'
import json
from pathlib import Path
p=Path('/tmp/mojolearn-transformer-transfer-apple');base=json.loads((p/'baseline.json').read_text())
for name in ('caller.2','baseline.2','baseline.3','caller.3'):
 row=json.loads((p/(name+'.json')).read_text());assert row['sha256']==base['sha256'];print(name,{k:v['median_ms'] for k,v in row['timings'].items()})
PY
