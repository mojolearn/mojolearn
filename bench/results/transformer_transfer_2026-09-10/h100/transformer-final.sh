#!/bin/bash
set -euo pipefail
if [ "${TF_FINAL_ACTIVE:-0}" != 1 ]; then
 export TF_FINAL_ACTIVE=1
 exec /root/.pixi/bin/pixi run --manifest-path /root/mojolearn/pixi.toml bash "$0"
fi
cd /root/mojolearn
out=/root/jobs/transformer-transfer
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python
sha256sum bindings/_mojolearn_transformer.mojo > "$out/default-source.sha256"
mojo build -j 2 --emit shared-lib --target-cpu x86-64-v3 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . -I bindings bindings/_mojolearn_transformer.mojo -o "$out/default.so" > "$out/build-default.log" 2>&1
cp "$out/default.so" python/mojolearn/identical/_mojolearn_transformer.so
/usr/bin/python3 tools/transformer_transfer_check.py --output "$out/default.json" > "$out/check-default.log" 2>&1
for arm in default baseline; do
 cp "$out/$arm.so" python/mojolearn/identical/_mojolearn_transformer.so
 MOJOLEARN_SPEED_DUMP_DIR="$out/dump-$arm-reverse" /usr/bin/python3 bench/speed/seq_py_speed_arm.py --lane transformer --rows large --rounds 7 > "$out/price-$arm-reverse.log" 2>&1
done
cp "$out/default.so" python/mojolearn/identical/_mojolearn_transformer.so
/usr/bin/python3 - "$out" <<'PY'
import json,hashlib,re,statistics,sys
from pathlib import Path
p=Path(sys.argv[1]);base=json.loads((p/'baseline.json').read_text());final=json.loads((p/'default.json').read_text())
assert base['sha256']==final['sha256'] and len(final['sha256'])==82
prices={}
for arm in ('default','baseline'):
 for f in (p/'dump-baseline').glob('*.bin'):
  assert hashlib.sha256(f.read_bytes()).digest()==hashlib.sha256((p/f'dump-{arm}-reverse'/f.name).read_bytes()).digest()
 rows={}
 for shape,ms in re.findall(r'^FSPEED lane=transformer arm=ours shape=(\S+) round=\d+ ms=(\S+)',(p/f'price-{arm}-reverse.log').read_text(),re.M):rows.setdefault(shape,[]).append(float(ms))
 assert len(rows)==2 and all(len(v)==7 for v in rows.values())
 prices[arm]={k:dict(samples_ms=v,median_ms=statistics.median(v)) for k,v in rows.items()}
(p/'default-summary.json').write_text(json.dumps(dict(arrays_equal=82,timings=prices),indent=2)+'\n')
print('DEFAULT NVIDIA TRANSFORMER PASS')
PY
