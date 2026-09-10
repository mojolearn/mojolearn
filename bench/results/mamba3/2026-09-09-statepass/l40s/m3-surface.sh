#!/bin/bash
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mamba3
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mamba3/python
out=/root/jobs/m3-finish
python3 python/mojolearn/tests/test_mamba_surface.py > "$out/surface.log" 2>&1
pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . mamba/checks/mamba_decode_check.mojo > "$out/mamba1-decode.log" 2>&1
