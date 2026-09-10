#!/bin/bash
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mamba3
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mamba3/python
out=/root/jobs/m3-h100
mkdir -p "$out" python/mojolearn/identical
pixi install > "$out/install.log" 2>&1
pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . mamba/checks/mamba3_refusal_check.mojo > "$out/refusal-kernel.log" 2>&1
pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . mamba/checks/mamba_decode_check.mojo > "$out/mamba1-decode.log" 2>&1
for arm in baseline optimized; do
 extra=()
 if [ "$arm" = baseline ]; then extra=(-D MOJOLEARN_MAMBA3_LEGACY_STATEPASS=1 -D MOJOLEARN_MAMBA3_LEGACY_HOST_COPY=1 -D MOJOLEARN_MAMBA3_LEGACY_ANGLE_INCREMENT=1 -D MOJOLEARN_MAMBA3_LEGACY_TRACE_SLICES=1 -D MOJOLEARN_MAMBA3_LEGACY_REFUSAL=1); fi
 pixi run mojo build -j 2 --emit shared-lib --target-cpu x86-64-v3 -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${extra[@]}" -I . -I bindings bindings/_mojolearn_mamba.mojo -o "$out/$arm.so" > "$out/build-$arm.log" 2>&1
 cp "$out/$arm.so" python/mojolearn/identical/_mojolearn_mamba.so
 MOJOLEARN_SPEED_DUMP_DIR="$out/dump-$arm" python3 bench/speed/seq_py_speed_arm.py --lane mamba3 --rounds 5 > "$out/price-$arm.log" 2>&1
done
python3 python/mojolearn/tests/test_mamba_surface.py > "$out/surface.log" 2>&1
python3 - <<'PY'
import hashlib,pathlib,json,re,statistics
out=pathlib.Path('/root/jobs/m3-h100');rows=[]
prior={r['file']:r for r in json.loads(pathlib.Path('/root/m3-l40s-identity.json').read_text())}
for f in sorted((out/'dump-baseline').glob('*.bin')):
 a=hashlib.sha256(f.read_bytes()).hexdigest();other=out/'dump-optimized'/f.name;b=hashlib.sha256(other.read_bytes()).hexdigest()
 rows.append(dict(file=f.name,bytes=f.stat().st_size,baseline_sha256=a,optimized_sha256=b,equal=a==b,l40s_equal=a==prior[f.name]['baseline_sha256']))
assert len(rows)==3 and all(r['equal'] and r['l40s_equal'] for r in rows),rows
(out/'full-output-identity.json').write_text(json.dumps(rows,indent=2)+'\n')
summary={}
for arm in ['baseline','optimized']:
 for shape,ms in re.findall(r'^FSPEED lane=mamba3 arm=ours shape=(\S+) round=\d+ ms=([\d.]+)',(out/f'price-{arm}.log').read_text(),re.M):summary.setdefault(shape,{}).setdefault(arm,[]).append(float(ms))
for shape,row in summary.items():
 row['baseline_median_ms']=statistics.median(row['baseline']);row['optimized_median_ms']=statistics.median(row['optimized']);row['candidate_over_baseline']=row['optimized_median_ms']/row['baseline_median_ms']
(out/'timings.json').write_text(json.dumps(summary,indent=2)+'\n')
PY
