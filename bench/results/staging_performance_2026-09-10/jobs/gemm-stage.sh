#!/bin/bash
set -euo pipefail
export PATH=/root/.pixi/bin:$PATH
for attempt in $(seq 1 90); do
 test -x /root/.pixi/bin/pixi && break
 sleep 2
done
cd /root/stageperf
if [ "${STAGE_PERF_ACTIVE:-0}" != 1 ]; then
 export STAGE_PERF_ACTIVE=1
 exec pixi run bash /root/jobs/gemm-stage.sh
fi
out=/root/evidence/gemm-stage
mkdir -p "$out"
mojo --version > "$out/compiler.txt"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$out/gpu.txt"
lscpu > "$out/cpu.txt"
sha256sum gemm/checks/gemm_identical.mojo > "$out/source-sha256.txt"
for arm in baseline staged; do
 extra=()
 if [ "$arm" = staged ]; then extra=(-D MOJOLEARN_GEMM_STAGE_FTZ=1); fi
 mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${extra[@]}" -I . bench/speed/gemm_speed_main.mojo -o "$out/speed-$arm" > "$out/build-speed-$arm.log" 2>&1
 if [ "$arm" = staged ]; then
 mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${extra[@]}" -I . gemm/checks/gemm_device_check.mojo -o "$out/device-$arm" > "$out/build-device-$arm.log" 2>&1
 "$out/device-$arm" > "$out/device-$arm.log" 2>&1
 fi
done
export MOJOLEARN_SPEED_GEMM_ARMS=v1 MOJOLEARN_SPEED_ROUNDS=7
export MOJOLEARN_SPEED_SHAPES=llama8b.qkv.t512,llama8b.mlp_up.t512,llama8b.mlp_down.t512,pca.transform.wide.8192x64x128,kmeans.dist.4096x64x64
for arm in baseline staged staged baseline; do
 trial=1
 if test -f "$out/price-$arm-1.log"; then trial=2; fi
 "$out/speed-$arm" > "$out/price-$arm-$trial.log" 2>&1
done
python3 - "$out" <<'PY'
import pathlib,sys,re,json,statistics
p=pathlib.Path(sys.argv[1]); rows={}; hashes={}
for arm in ['baseline','staged']:
 for trial in [1,2]:
  samples={}
  for shape,ms,h in re.findall(r'^FSPEED lane=gemm arm=\S+ shape=(\S+) round=\d+ ms=(\S+) hash=(\S+)',(p/f'price-{arm}-{trial}.log').read_text(),re.M):
   samples.setdefault(shape,[]).append(float(ms)); hashes.setdefault(shape,set()).add(h)
  assert len(samples)==5 and all(len(v)==7 for v in samples.values())
  rows[f'{arm}-{trial}']={k:dict(samples_ms=v,median_ms=statistics.median(v)) for k,v in samples.items()}
assert all(len(v)==1 for v in hashes.values())
(p/'summary.json').write_text(json.dumps(dict(timings=rows,hashes={k:list(v)[0] for k,v in hashes.items()}),indent=2)+'\n')
PY
printf 'PASS\n' > "$out/verdict.txt"
