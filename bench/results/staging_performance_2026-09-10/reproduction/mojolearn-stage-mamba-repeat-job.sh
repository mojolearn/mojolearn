#!/bin/bash
set -euo pipefail
export PATH=/root/.pixi/bin:$PATH
cd /root/finalstage
while [ ! -f /root/jobs/knn-public.rc ]; do sleep 3; done
if [ "${MAMBA_REPEAT_ACTIVE:-0}" != 1 ]; then
 export MAMBA_REPEAT_ACTIVE=1
 exec pixi run bash /root/jobs/mamba-repeat.sh
fi
out=/root/evidence/mamba-repeat
mkdir -p "$out"
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/finalstage/python
sha256sum /root/evidence/mamba-stage/baseline.so /root/evidence/final-stage/mamba-baseline.so > "$out/binary-sha256.txt"
nvidia-smi -q > "$out/gpu-before.txt"
ps -eo pid,etime,args > "$out/processes-before.txt"
for arm in baseline default baseline; do
 trial=0
 if test -f "$out/price-$arm-0.log"; then trial=1; fi
 cp "/root/evidence/final-stage/mamba-$arm.so" python/mojolearn/identical/_mojolearn_mamba.so
 MOJOLEARN_SPEED_DUMP_DIR="$out/dump-$arm-$trial" /usr/bin/python3 bench/speed/seq_py_speed_arm.py --lane mamba3 --rounds 5 > "$out/price-$arm-$trial.log" 2>&1
done
nvidia-smi -q > "$out/gpu-after.txt"
/usr/bin/python3 - "$out" <<'PY'
import pathlib,json,hashlib,re,statistics,sys
p=pathlib.Path(sys.argv[1]); hashes={};timings={}
for arm,trial in [('baseline',0),('default',0),('baseline',1)]:
 files=sorted((p/f'dump-{arm}-{trial}').glob('*.bin'));assert len(files)==3
 for f in files:hashes.setdefault(f.name,set()).add(hashlib.sha256(f.read_bytes()).hexdigest())
 samples={}
 for shape,ms in re.findall(r'^FSPEED lane=\S+ arm=ours shape=(\S+) round=\d+ ms=(\S+)',(p/f'price-{arm}-{trial}.log').read_text(),re.M):samples.setdefault(shape,[]).append(float(ms))
 assert len(samples)==3 and all(len(v)==5 for v in samples.values())
 timings[f'{arm}-{trial}']={k:dict(samples_ms=v,median_ms=statistics.median(v)) for k,v in samples.items()}
assert all(len(v)==1 for v in hashes.values())
(p/'summary.json').write_text(json.dumps(dict(hashes={k:list(v)[0] for k,v in hashes.items()},timings=timings),indent=2)+'\n')
PY
