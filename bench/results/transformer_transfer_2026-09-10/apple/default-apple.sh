#!/bin/bash
set -euo pipefail
cd /Users/andrewhendel/CascadeProjects/mojolearn
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python
bash bindings/build_transformer.sh
python tools/transformer_transfer_check.py --output /tmp/mojolearn-transformer-transfer-apple/default.json
python - <<'PY'
import json
from pathlib import Path
p=Path('/tmp/mojolearn-transformer-transfer-apple')
a=json.loads((p/'baseline.json').read_text());b=json.loads((p/'default.json').read_text())
assert a['sha256']==b['sha256'] and len(b['sha256'])==82
print('DEFAULT APPLE TRANSFORMER PASS: 82 complete arrays match legacy')
PY
