#!/bin/bash
set -euo pipefail
cd /Users/andrewhendel/CascadeProjects/mojolearn
out=/tmp/mojolearn-m3-default-apple
mkdir -p "$out" python/mojolearn/identical
pixi run mojo build -j 2 --emit shared-lib -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . -I bindings bindings/_mojolearn_mamba.mojo -o python/mojolearn/identical/_mojolearn_mamba.so > "$out/build.log" 2>&1
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python
pixi run python tools/mamba3_fresh_prefill_check.py > "$out/fresh.log" 2>&1
pixi run python python/mojolearn/tests/test_mamba_surface.py > "$out/surface.log" 2>&1
