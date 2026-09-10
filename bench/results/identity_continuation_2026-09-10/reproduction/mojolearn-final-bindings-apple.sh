#!/bin/bash
set -euo pipefail
cd /Users/andrewhendel/CascadeProjects/mojolearn
out=bench/results/identity_continuation_2026-09-10
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python
sh bindings/build_transformer.sh > "$out/transformer-final-build-apple.log" 2>&1
pixi run python tools/transformer_transfer_check.py --output "$out/transformer-final-apple.json" --rounds 3 > "$out/transformer-final-apple.log" 2>&1
pixi run python - <<'PY'
import json
from pathlib import Path
p=Path('bench/results/identity_continuation_2026-09-10/transformer-final-apple.json');a=json.loads(p.read_text());b=json.loads(Path('bench/results/transformer_fresh_2026-09-10/apple/baseline.json').read_text());assert a['sha256']==b['sha256'];print('Final Apple transformer PASS82 unchanged')
PY
sh bindings/build_estimators.sh > "$out/pca-final-build-apple.log" 2>&1
pixi run python tools/pca_wide_surface_check.py > "$out/pca-final-public-apple.log" 2>&1
