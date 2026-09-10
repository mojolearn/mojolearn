#!/bin/bash
set -euo pipefail
if [ "${TF_TRANSFER_ACTIVE:-0}" != 1 ]; then
 export TF_TRANSFER_ACTIVE=1
 exec /root/.pixi/bin/pixi run --manifest-path /root/mojolearn/pixi.toml bash "$0"
fi
cd /root/mojolearn
out=/root/jobs/transformer-transfer
mkdir -p "$out" python/mojolearn/identical bench/speed
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python
mojo --version > "$out/compiler.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv > "$out/gpu.txt"
/usr/bin/python3 -c 'import numpy,sys; print(sys.version,numpy.__version__)' > "$out/python.txt"
sha256sum bindings/_mojolearn_transformer.mojo tools/transformer_transfer_check.py > "$out/sources.sha256"
mojo run -I . checks/portable_log2_64_check.mojo > "$out/log2.log" 2>&1
for arm in baseline caller; do
 extra=()
 if [ "$arm" = caller ]; then extra=(-D MOJOLEARN_TRANSFORMER_CALLER_TRANSFER=1); fi
 mojo build -j 2 --emit shared-lib --target-cpu x86-64-v3 -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${extra[@]}" -I . -I bindings bindings/_mojolearn_transformer.mojo -o "$out/$arm.so" > "$out/build-$arm.log" 2>&1
 cp "$out/$arm.so" python/mojolearn/identical/_mojolearn_transformer.so
 /usr/bin/python3 tools/transformer_transfer_check.py --output "$out/$arm.json" > "$out/check-$arm.log" 2>&1
 /usr/bin/python3 python/mojolearn/tests/test_transformer_surface.py > "$out/surface-$arm.log" 2>&1
 /usr/bin/python3 python/mojolearn/tests/test_transformer_hd128.py > "$out/hd128-$arm.log" 2>&1
 MOJOLEARN_SPEED_DUMP_DIR="$out/dump-$arm" /usr/bin/python3 bench/speed/seq_py_speed_arm.py --lane transformer --rows large --rounds 7 > "$out/price-$arm.log" 2>&1
done
/usr/bin/python3 - "$out" <<'PY'
import hashlib,json,re,statistics,sys
from pathlib import Path
p=Path(sys.argv[1]);a=json.loads((p/'baseline.json').read_text());b=json.loads((p/'caller.json').read_text())
assert a['sha256']==b['sha256'] and len(a['sha256'])==82
rows=[]
for f in sorted((p/'dump-baseline').glob('*.bin')):
 sha=hashlib.sha256(f.read_bytes()).hexdigest()
 actual=hashlib.sha256((p/'dump-caller'/f.name).read_bytes()).hexdigest()
 assert sha==actual
 rows.append(dict(file=f.name,bytes=f.stat().st_size,sha256=sha))
assert len(rows)==2
prices={}
for arm in ('baseline','caller'):
 samples={}
 for shape,ms in re.findall(r'^FSPEED lane=transformer arm=ours shape=(\S+) round=\d+ ms=(\S+)',(p/f'price-{arm}.log').read_text(),re.M):samples.setdefault(shape,[]).append(float(ms))
 assert len(samples)==2
 prices[arm]={k:dict(samples_ms=v,median_ms=statistics.median(v)) for k,v in samples.items()}
(p/'summary.json').write_text(json.dumps(dict(arrays_equal=82,large_outputs=rows,timings=prices),indent=2)+'\n')
PY
