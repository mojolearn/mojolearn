#!/bin/bash
set -euo pipefail
if [ "${M3_ENV_ACTIVE:-0}" != 1 ]; then
 export M3_ENV_ACTIVE=1
 exec /root/.pixi/bin/pixi run --manifest-path /root/mojolearn/pixi.toml bash "$0"
fi
cd /root/mamba3-residual
export PYTHONPATH=/root/mamba3-residual/python MOJOLEARN_NUMERIC_MODE=identical
out=/root/jobs/m3-fresh-final
python3 -c "import sys,numpy; print(sys.version, numpy.__version__, sys.executable)" > "$out/python-runtime.txt"
cp "$out/default.so" python/mojolearn/identical/_mojolearn_mamba.so
MOJOLEARN_MAMBA3_FRESH_LARGE=1 python3 tools/mamba3_fresh_prefill_check.py > "$out/fresh-default.log" 2>&1
python3 python/mojolearn/tests/test_mamba_surface.py > "$out/surface-default.log" 2>&1
MOJOLEARN_SPEED_DUMP_DIR="$out/dump-default" python3 bench/speed/seq_py_speed_arm.py --lane mamba3 --rounds 5 > "$out/price-default.log" 2>&1
python3 - <<'PY'
from pathlib import Path
import hashlib,json,re,statistics
out=Path('/root/jobs/m3-fresh-final');identities=[];timings={}
for f in sorted(Path('/root/jobs/m3-fresh/dump-baseline').glob('*.bin')):
 base=hashlib.sha256(f.read_bytes()).hexdigest();actual=hashlib.sha256((out/'dump-default'/f.name).read_bytes()).hexdigest()
 identities.append(dict(file=f.name,bytes=f.stat().st_size,baseline_sha256=base,actual_sha256=actual,equal=base==actual))
assert len(identities)==3 and all(r['equal'] for r in identities)
for shape,ms in re.findall(r'^FSPEED lane=mamba3 arm=ours shape=(\S+) round=\d+ ms=(\S+)',(out/'price-default.log').read_text(),re.M):timings.setdefault(shape,[]).append(float(ms))
(out/'full-output-identity.json').write_text(json.dumps(identities,indent=2)+'\n')
(out/'timings.json').write_text(json.dumps({k:dict(samples_ms=v,median_ms=statistics.median(v)) for k,v in timings.items()},indent=2)+'\n')
PY
