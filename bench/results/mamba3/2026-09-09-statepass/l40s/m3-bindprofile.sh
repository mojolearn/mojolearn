#!/bin/bash
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mamba3
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mamba3/python
out=/root/jobs/m3-bindprofile
mkdir -p "$out"
pixi run mojo build -j 2 --emit shared-lib --target-cpu x86-64-v3 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_MAMBA3_PHASE_TIMERS=1 -I . -I bindings bindings/_mojolearn_mamba.mojo -o "$out/profile.so" > "$out/build.log" 2>&1
cp "$out/profile.so" python/mojolearn/identical/_mojolearn_mamba.so
python3 bench/speed/seq_py_speed_arm.py --lane mamba3 --rounds 1 > "$out/profile.log" 2>&1
