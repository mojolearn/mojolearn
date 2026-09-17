#!/bin/bash
# Fresh CPU clean/negative controls and public reference replay for promotion.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MOJOLEARN_NUMERIC_MODE=identical
PY=.pixi/envs/test/bin/python
LANES=spectral,metrics-fowlkes-mallows,arima-exog,arima-exog-seasonal
export MOJOLEARN_HOST_DIR="$PWD/python/mojolearn/host"
# Both arms share production core helpers.
CORE="$MOJOLEARN_HOST_DIR/_mojolearn_core_host.so"
if [[ ! -f "$CORE" ]]; then
 echo "Missing core host dependency: run with --build core,metrics,arima" >&2
 exit 1
fi
cp "$CORE" "$PWD/python/mojolearn/host-sabotage/"
"$PY" tools/identity_break.py --lanes "$LANES" --repeats 2 --json "$LEG_OUT/cpu-clean.json" > "$LEG_OUT/clean.log" 2>&1
MOJOLEARN_HOST_DIR="$PWD/python/mojolearn/host-sabotage" MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
 "$PY" tools/identity_break.py --lanes "$LANES" --repeats 2 --json "$LEG_OUT/cpu-sabotage.json" > "$LEG_OUT/sabotage.log" 2>&1
"$PY" - "$LEG_OUT" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,'tools')
import verification_matrix as matrix
out=Path(sys.argv[1]);clean=json.loads((out/'cpu-clean.json').read_text());bad=json.loads((out/'cpu-sabotage.json').read_text())
assert clean['commit']==bad['commit']
for family in ('_mojolearn_metrics_host','_mojolearn_arima_host'):
 assert not clean['host']['families'][family]['sabotage']
 assert bad['host']['families'][family]['sabotage']
rows=[]
for key,c in clean['cells'].items():
 b=bad['cells'][key]
 rows.append(dict(cell=key,clean=matrix.stable_digest(c,'train'),sabotage=matrix.stable_digest(b,'train'),detected=matrix.negative_control_moves(b,c,'train')))
result=dict(cells=rows,expected_cells=36,passed=len(rows)==36 and all(r['detected'] for r in rows))
(out/'negative-controls.json').write_text(json.dumps(result,indent=2)+'\n')
assert result['passed'],result
PY
PYTHONPATH="$PWD/python" "$PY" -m mojolearn verify --lanes "$LANES" --repeats 2 --no-models --json > "$LEG_OUT/public-clean.json" 2> "$LEG_OUT/public-clean.log"
