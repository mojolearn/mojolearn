#!/bin/bash
set -euo pipefail
cd /Users/andrewhendel/CascadeProjects/mojolearn
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . mamba/checks/mamba3_check.mojo -o /tmp/mojolearn-mamba3-default-final > /tmp/mojolearn-mamba3-default-final-build.log 2>&1
bash /tmp/mojolearn-mamba3-default-final-check.sh
MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_mamba.sh > /tmp/mojolearn-mamba3-default-final-binding.log 2>&1
cd python
MOJOLEARN_NUMERIC_MODE=identical pixi run python -m mojolearn.tests.test_mamba_surface > /tmp/mojolearn-mamba3-default-final-surface.log 2>&1
