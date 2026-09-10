#!/bin/bash
set -euo pipefail
if [ "${M3_ENV_ACTIVE:-0}" != 1 ]; then
  export M3_ENV_ACTIVE=1
  exec /root/.pixi/bin/pixi run --manifest-path /root/mojolearn/pixi.toml bash "$0"
fi
cd /root/mamba3-next
out=/root/jobs/m3-increment-final
mkdir -p "$out"
sha256sum mamba/impl/mamba_ssm/ops/mamba3_siso.mojo gemm/checks/gemm_identical.mojo > "$out/source-sha256.txt"
mojo build --emit shared-lib -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . -I bindings bindings/_mojolearn_mamba.mojo -o "$out/default.so" > "$out/build.log" 2>&1
cp "$out/default.so" python/mojolearn/identical/_mojolearn_mamba.so
export PYTHONPATH=/root/mamba3-next/python MOJOLEARN_NUMERIC_MODE=identical
MOJOLEARN_MAMBA3_FRESH_LARGE=1 /usr/bin/python3 tools/mamba3_fresh_prefill_check.py > "$out/fresh.log" 2>&1
/usr/bin/python3 python/mojolearn/tests/test_mamba_surface.py > "$out/surface.log" 2>&1
MOJOLEARN_SPEED_DUMP_DIR="$out/dump-default" /usr/bin/python3 bench/speed/seq_py_speed_arm.py --lane mamba3 --rounds 5 > "$out/price.log" 2>&1
/usr/bin/python3 - "$out" <<'PY'
from pathlib import Path
import hashlib,json,re,statistics,sys
out=Path(sys.argv[1]); rows=[]; samples={}
for f in sorted(Path('/root/jobs/m3-increment-tile/dump-baseline').glob('*.bin')):
 a=hashlib.sha256(f.read_bytes()).hexdigest(); b=hashlib.sha256((out/'dump-default'/f.name).read_bytes()).hexdigest()
 rows.append(dict(file=f.name,bytes=f.stat().st_size,baseline_sha256=a,default_sha256=b,equal=a==b))
assert len(rows)==3 and all(r['equal'] for r in rows)
for shape,ms in re.findall(r'^FSPEED lane=mamba3 arm=ours shape=(\S+) round=\d+ ms=(\S+)',(out/'price.log').read_text(),re.M): samples.setdefault(shape,[]).append(float(ms))
(out/'full-output-identity.json').write_text(json.dumps(rows,indent=2)+'\n')
(out/'timings.json').write_text(json.dumps({k:dict(samples_ms=v,median_ms=statistics.median(v)) for k,v in samples.items()},indent=2)+'\n')
PY
