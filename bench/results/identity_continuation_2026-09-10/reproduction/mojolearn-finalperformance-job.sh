#!/bin/bash
set -euo pipefail
export PATH=/root/.pixi/bin:$PATH
cd /root/finalperformance
while [ ! -f /root/jobs/identity-device-final.rc ]; do sleep 3; done
if [ "${FINAL_PERF_ACTIVE:-0}" != 1 ]; then
 export FINAL_PERF_ACTIVE=1
 exec pixi run bash /root/jobs/final-performance.sh
fi
out=/root/evidence/final-performance
mkdir -p "$out" python/mojolearn/identical bench/speed transformer/corpus
tar -xzf /root/mamba-fixtures.tar.gz
cp /root/transformer-gen-corpus.py transformer/corpus/gen_corpus.py
cp bench/results/mamba3/2026-09-09-statepass/reproduction/seq_py_speed_arm.py bench/speed/
cp bench/results/mamba3/2026-09-09-statepass/reproduction/speed_torch_seq.py tools/
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/finalperformance/python
mojo --version > "$out/compiler.txt"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$out/gpu.txt"
lscpu > "$out/cpu.txt"
sha256sum bindings/_mojolearn_transformer.mojo mamba/impl/mamba_ssm/ops/mamba3_siso.mojo decomposition/impl/linalg/detail/svd_full.mojo checks/numerics.mojo > "$out/source-sha256.txt"
mojo build --emit shared-lib -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . -I bindings bindings/_mojolearn_mamba.mojo -o "$out/mamba-default.so" > "$out/build-mamba.log" 2>&1
for arm in default baseline; do
 lib="$out/mamba-default.so"
 if [ "$arm" = baseline ]; then lib=/root/evidence/mamba-yintra/baseline.so; fi
 cp "$lib" python/mojolearn/identical/_mojolearn_mamba.so
 MOJOLEARN_MAMBA3_FRESH_LARGE=1 /usr/bin/python3 tools/mamba3_fresh_prefill_check.py > "$out/mamba-check-$arm.log" 2>&1
 /usr/bin/python3 python/mojolearn/tests/test_mamba_surface.py > "$out/mamba-surface-$arm.log" 2>&1
 MOJOLEARN_SPEED_DUMP_DIR="$out/mamba-dump-$arm" /usr/bin/python3 bench/speed/seq_py_speed_arm.py --lane mamba3 --rounds 5 > "$out/mamba-price-$arm.log" 2>&1
done
mojo build -j 2 --emit shared-lib --target-cpu x86-64-v3 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . -I bindings bindings/_mojolearn_transformer.mojo -o "$out/transformer-default.so" > "$out/build-transformer.log" 2>&1
for arm in default baseline; do
 lib="$out/transformer-default.so"
 if [ "$arm" = baseline ]; then lib=/root/evidence/transformer-fresh/baseline.so; fi
 cp "$lib" python/mojolearn/identical/_mojolearn_transformer.so
 /usr/bin/python3 tools/transformer_transfer_check.py --output "$out/transformer-$arm.json" > "$out/transformer-check-$arm.log" 2>&1
 if [ "$arm" = default ]; then /usr/bin/python3 tools/transformer_fresh_prefill_check.py > "$out/transformer-fresh.log" 2>&1; fi
 MOJOLEARN_SPEED_DUMP_DIR="$out/transformer-dump-$arm" /usr/bin/python3 bench/speed/seq_py_speed_arm.py --lane transformer --rows large --rounds 7 > "$out/transformer-price-$arm.log" 2>&1
done
/usr/bin/python3 - "$out" <<'PY'
from pathlib import Path
import hashlib,json,re,statistics,sys
p=Path(sys.argv[1]); result={}
for lane in ['mamba','transformer']:
 outputs=[];timings={}
 files=sorted((p/f'{lane}-dump-default').glob('*.bin'))
 assert len(files)==(3 if lane=='mamba' else 2)
 for f in files:
  a=hashlib.sha256(f.read_bytes()).hexdigest();b=hashlib.sha256((p/f'{lane}-dump-baseline'/f.name).read_bytes()).hexdigest()
  assert a==b
  outputs.append(dict(file=f.name,bytes=f.stat().st_size,sha256=a))
 for arm in ['default','baseline']:
  samples={}
  for shape,ms in re.findall(r'^FSPEED lane=\S+ arm=ours shape=(\S+) round=\d+ ms=(\S+)',(p/f'{lane}-price-{arm}.log').read_text(),re.M):samples.setdefault(shape,[]).append(float(ms))
  timings[arm]={k:dict(samples_ms=v,median_ms=statistics.median(v)) for k,v in samples.items()}
 result[lane]=dict(outputs=outputs,timings=timings)
a=json.loads((p/'transformer-default.json').read_text());b=json.loads((p/'transformer-baseline.json').read_text());assert a['sha256']==b['sha256'];result['arrays_equal']=len(a['sha256'])
(p/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
PY
mojo build -j 2 --emit shared-lib --target-cpu x86-64-v3 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . -I bindings bindings/_mojolearn_estimators.mojo -o python/mojolearn/identical/_mojolearn_estimators.so > "$out/build-estimators.log" 2>&1
/usr/bin/python3 tools/pca_wide_surface_check.py > "$out/pca-public.log" 2>&1
printf 'PASS\n' > "$out/verdict.txt"
