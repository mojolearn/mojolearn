#!/bin/sh
set -eu
cd /root/iforest-pool-out
python3 - <<'PY' > comparison.log
import json
from pathlib import Path
for name in ('iforest','svm'):
    actual=json.loads(Path(name+'.json').read_text())
    expected=json.loads(Path('reference-'+name+'.json').read_text())
    assert actual == expected,name
print('PASS full prior IsolationForest and SVM receipt equality')
PY
