#!/bin/bash
set -euo pipefail
cd /Users/andrewhendel/mojolearn-evidence/cpu-verification-completion/wheel-stage
/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/pkg/bin/python setup.py bdist_wheel --dist-dir ../dist > ../host-export-wheel-build.log 2>&1
../installed-env/bin/python -m pip install --no-deps --force-reinstall ../dist/*.whl > ../host-export-wheel-install.log 2>&1
cd ..
installed-env/bin/python audit_installed_exports.py > installed-exports-rebuilt-audit.json
installed-env/bin/python check_installed.py gbdt-adapter-score-weighted,rf-score-weighted,gbdt-nan-modes,gbdt-parametric-losses,gbdt-lossguide-newtoncosine,gbdt-pair-logit,gbdt-yeti-rank host-export
