#!/bin/bash
set -euo pipefail
root=/Users/andrewhendel/CascadeProjects/mojolearn
candidate=/tmp/mojolearn-mamba3-residual
out=/tmp/mojolearn-m3-residual-apple
mkdir -p "$out" "$candidate/python/mojolearn/identical"
cd "$candidate"
"$root/.pixi/envs/default/bin/mojo" --version > "$out/version.log" 2>&1
pixi run --manifest-path "$root/pixi.toml" mojo build -j 2 --emit shared-lib -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_MAMBA3_FRESH_PREFILL=1 -D MOJOLEARN_MAMBA3_CALLER_TRANSFER=1 -I "$candidate" -I "$candidate/bindings" "$candidate/bindings/_mojolearn_mamba.mojo" -o "$candidate/python/mojolearn/identical/_mojolearn_mamba.so" > "$out/build.log" 2>&1
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$candidate/python" pixi run --manifest-path "$root/pixi.toml" python "$candidate/tools/mamba3_fresh_prefill_check.py" > "$out/fresh.log" 2>&1
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$candidate/python" pixi run --manifest-path "$root/pixi.toml" python "$candidate/python/mojolearn/tests/test_mamba_surface.py" > "$out/surface.log" 2>&1
