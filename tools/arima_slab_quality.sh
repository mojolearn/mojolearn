#!/bin/bash
# ARIMA_SLAB full quality (lane w3-arima): staged A/B arms, no builds;
# byte-identical orders/IC/d/params/llf/n_iter/retcode/predictions/forecasts
# on 512/2048-obs 4-series fixtures and a 48-series 1392-obs board-shaped one.
# Usage: bash tools/arima_slab_quality.sh <quality-tag>
set -euo pipefail
arima_tag=${1:?quality tag required}
[[ "$arima_tag" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
cd "$(dirname "$0")/.."
arima_python=$HOME/board-0834/cache/venv/bin/python
arima_out=$HOME/mq/out/arima-quality-$arima_tag
# Existing capture helper refuses artifact reuse and restores installed .so.
ARIMA_QUALITY_SMALL=0 ARIMA_QUALITY_WIDE=1 bash tools/arima_fast_quality_ab.sh "$arima_tag"
"$arima_python" - "$arima_tag" <<'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
tag = sys.argv[1]
out = Path.home() / 'mq/out' / ('arima-quality-' + tag)
# Reject a successful small-only run, stale capture, or wrong installed arm.
hashes = {}
for arm in ('A', 'B'):
    so = Path.home() / 'afc-def' / tag / (arm + '.so')
    hashes[arm] = hashlib.sha256(so.read_bytes()).hexdigest()
    records = [json.loads(line.split(' ', 1)[1]) for line in (out / (arm + '.log')).read_text().splitlines()
               if line.startswith('ARIMA_FAST_CAPTURE ')]
    assert len(records) == 1 and records[0]['binding_sha256'] == hashes[arm]
    with np.load(out / (arm + '.npz'), allow_pickle=False) as arrays:
        assert {key for key in arrays.files if key.endswith('/input')} == {'seed23_n512/input', 'seed87_n2048/input', 'seed61_n1392/input'}
        for prefix in ('seed23_n512', 'seed87_n2048', 'seed61_n1392'):
            for suffix in ('order_', 'ic_', 'd_', 'prediction', 'forecast'):
                assert prefix + '/' + suffix in arrays.files
assert 'ARIMA_FAST_QUALITY status=PASS' in (out / 'compare.log').read_text()
receipt = dict(source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               hashes=hashes, quality_tag=tag, fixtures=['seed23_n512', 'seed87_n2048', 'seed61_n1392'], status='PASS')
# Created only after full paired output comparison and fixture/hash checks.
with (out / 'FULL_PASS.json').open('x') as stream:
    json.dump(receipt, stream, sort_keys=True)
print('ARIMA_SLAB_FULL_QUALITY status=PASS tag=' + tag + ' source=' + receipt['source_sha'])
PY
