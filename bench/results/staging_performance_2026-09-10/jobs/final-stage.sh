#!/bin/bash
set -euo pipefail
export PATH=/root/.pixi/bin:$PATH
cd /root/finalstage
while [ ! -f /root/jobs/knn-batch.rc ]; do sleep 3; done
if [ "${FINAL_STAGE_ACTIVE:-0}" != 1 ]; then
 export FINAL_STAGE_ACTIVE=1
 exec pixi run bash /root/jobs/final-stage.sh
fi
out=/root/evidence/final-stage
mkdir -p "$out" python/mojolearn/identical transformer/corpus bench/speed
tar xzf /root/stage-mamba-fixtures.tar.gz
tar xzf /root/stage-grid-real.tar.gz
cp /root/stage-transformer-gen.py transformer/corpus/gen_corpus.py
cp bench/results/mamba3/2026-09-09-statepass/reproduction/seq_py_speed_arm.py bench/speed/
cp bench/results/mamba3/2026-09-09-statepass/reproduction/speed_torch_seq.py tools/
export PYTHONPATH=/root/finalstage/python MOJOLEARN_NUMERIC_MODE=identical
python=/usr/bin/python3
mojo --version > "$out/compiler.txt"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$out/gpu.txt"
sha256sum gemm/checks/gemm_identical.mojo mamba/impl/mamba_ssm/modules/mamba3.mojo bindings/_mojolearn_mamba.mojo bindings/_mojolearn_transformer.mojo > "$out/source-sha256.txt"
for lane in mamba transformer; do
 mojo build --emit shared-lib -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . -I bindings "bindings/_mojolearn_$lane.mojo" -o "$out/$lane-default.so" > "$out/build-$lane-default.log" 2>&1
 if [ "$lane" = mamba ]; then
  cp /root/evidence/mamba-stage/baseline.so "$out/mamba-baseline.so"
 else
  mojo build --emit shared-lib -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_LEGACY_STAGE_FTZ=1 -I . -I bindings bindings/_mojolearn_transformer.mojo -o "$out/transformer-baseline.so" > "$out/build-transformer-baseline.log" 2>&1
 fi
 for arm in default baseline; do
  cp "$out/$lane-$arm.so" "python/mojolearn/identical/_mojolearn_$lane.so"
  if [ "$lane" = mamba ]; then
   MOJOLEARN_MAMBA3_FRESH_LARGE=1 "$python" tools/mamba3_fresh_prefill_check.py > "$out/mamba-check-$arm.log" 2>&1
   "$python" python/mojolearn/tests/test_mamba_surface.py > "$out/mamba-surface-$arm.log" 2>&1
  else
   "$python" tools/transformer_transfer_check.py --output "$out/transformer-$arm.json" > "$out/transformer-check-$arm.log" 2>&1
   "$python" tools/transformer_fresh_prefill_check.py > "$out/transformer-fresh-$arm.log" 2>&1
  fi
 done
 for pass in 0 1; do
  arms=(default baseline)
  if [ "$pass" = 1 ]; then arms=(baseline default); fi
  for arm in "${arms[@]}"; do
   cp "$out/$lane-$arm.so" "python/mojolearn/identical/_mojolearn_$lane.so"
   price_lane=$lane
   extra=()
   rounds=7
   if [ "$lane" = mamba ]; then price_lane=mamba3; rounds=5; else extra=(--rows large); fi
   MOJOLEARN_SPEED_DUMP_DIR="$out/$lane-dump-$arm-$pass" "$python" bench/speed/seq_py_speed_arm.py --lane "$price_lane" "${extra[@]}" --rounds "$rounds" > "$out/$lane-price-$arm-$pass.log" 2>&1
  done
 done
done
"$python" - "$out" <<'PY'
import pathlib,json,hashlib,re,statistics,sys
p=pathlib.Path(sys.argv[1]); result={}
for lane,count,rounds in [('mamba',3,5),('transformer',2,7)]:
 hashes={}; timings={}
 for arm in ['default','baseline']:
  for trial in [0,1]:
   files=sorted((p/f'{lane}-dump-{arm}-{trial}').glob('*.bin'));assert len(files)==count
   for f in files:hashes.setdefault(f.name,set()).add(hashlib.sha256(f.read_bytes()).hexdigest())
   samples={}
   for shape,ms in re.findall(r'^FSPEED lane=\S+ arm=ours shape=(\S+) round=\d+ ms=(\S+)',(p/f'{lane}-price-{arm}-{trial}.log').read_text(),re.M):samples.setdefault(shape,[]).append(float(ms))
   assert len(samples)==count and all(len(v)==rounds for v in samples.values())
   timings[f'{arm}-{trial}']={k:dict(samples_ms=v,median_ms=statistics.median(v)) for k,v in samples.items()}
 assert all(len(v)==1 for v in hashes.values())
 result[lane]=dict(hashes={k:list(v)[0] for k,v in hashes.items()},timings=timings)
a=json.loads((p/'transformer-default.json').read_text());b=json.loads((p/'transformer-baseline.json').read_text());assert a['sha256']==b['sha256'];result['transformer_exact_arrays']=len(a['sha256'])
(p/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
PY
printf 'PASS\n' > "$out/verdict.txt"
