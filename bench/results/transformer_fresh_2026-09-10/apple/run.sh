#!/bin/bash
set -euo pipefail
cd /tmp/mojolearn-mamba3-yintra
out=/tmp/mojolearn-transformer-fresh-apple
mkdir -p "$out" python/mojolearn/identical
pixi_root=/Users/andrewhendel/CascadeProjects/mojolearn/pixi.toml
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/tmp/mojolearn-mamba3-yintra/python
for arm in baseline fresh; do
  extra=()
  if [ "$arm" = fresh ]; then extra=(-D MOJOLEARN_TRANSFORMER_FRESH_PREFILL=1); fi
  pixi run --manifest-path "$pixi_root" mojo build -j 2 --emit shared-lib -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${extra[@]}" -I . -I bindings bindings/_mojolearn_transformer.mojo -o "$out/$arm.so" > "$out/build-$arm.log" 2>&1
  cp "$out/$arm.so" python/mojolearn/identical/_mojolearn_transformer.so
  pixi run --manifest-path "$pixi_root" python tools/transformer_transfer_check.py --output "$out/$arm.json" --rounds 3 > "$out/transfer-$arm.log" 2>&1
  if [ "$arm" = fresh ]; then
    pixi run --manifest-path "$pixi_root" python tools/transformer_fresh_prefill_check.py > "$out/fresh.log" 2>&1
    pixi run --manifest-path "$pixi_root" python python/mojolearn/tests/test_transformer_surface.py > "$out/surface.log" 2>&1
    pixi run --manifest-path "$pixi_root" python python/mojolearn/tests/test_transformer_hd128.py > "$out/hd128.log" 2>&1
  fi
done
pixi run --manifest-path "$pixi_root" python - "$out" <<'PY'
from pathlib import Path
import json,sys
p=Path(sys.argv[1]);a=json.loads((p/'baseline.json').read_text());b=json.loads((p/'fresh.json').read_text())
assert a['sha256']==b['sha256']
print('TF_FRESH_TRANSFER_PASS',len(a['sha256']))
PY
