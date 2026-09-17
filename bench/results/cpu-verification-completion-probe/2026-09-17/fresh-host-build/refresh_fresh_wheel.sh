#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME
cd /Users/andrewhendel/mojolearn-evidence/cpu-verification-completion/fresh-wheel-stage
/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/pkg/bin/python setup.py bdist_wheel --dist-dir ../fresh-dist > ../fresh-host-wheel-numpy-fix-build.log 2>&1
../fresh-installed-env/bin/python -m pip install --no-deps --force-reinstall ../fresh-dist/*.whl > ../fresh-host-wheel-numpy-fix-install.log 2>&1
cd ..
set +e
fresh-installed-env/bin/python -m mojolearn verify --coverage --json > missing-numpy-refusal.json 2> missing-numpy-refusal.log
result=$?
set -e
python3 - "$result" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path('missing-numpy-refusal.json').read_text())
assert int(sys.argv[1])!=0
assert 'CANNOT RUN' in str(r) and 'python -m pip install numpy' in str(r),r
assert 'Traceback' not in Path('missing-numpy-refusal.log').read_text()
print('Installed wheel gives actionable missing-NumPy refusal')
PY
fresh-installed-env/bin/python -m pip install 'numpy==2.5.3' > fresh-host-numpy-install.log 2>&1
fresh-installed-env/bin/python audit_installed_exports.py > fresh-host-exports.json
