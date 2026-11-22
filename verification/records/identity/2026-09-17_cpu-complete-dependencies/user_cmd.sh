#!/bin/bash
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MOJOLEARN_CPU_THREADS=1
.pixi/envs/test/bin/python - <<'PYBODY'
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os, subprocess, json, shutil
root=Path.cwd(); out=Path(os.environ['LEG_OUT'])/'records'; out.mkdir(parents=True,exist_ok=True)
prod=root/'python/mojolearn/host'; bad=root/'python/mojolearn/host-sabotage'
for src in prod.glob('*.so'):
 if not (bad/src.name).exists():shutil.copy2(src,bad/src.name)
lanes='transformer-bf16w,transformer-int8w,mamba1-bf16w,mamba1-int8w,mamba2-bf16w,mamba2-int8w,mamba3-bf16w,mamba3-int8w,mlp-bf16w,mlp-int8w,samba-bf16w,samba-int8w,gp-optimize,gp-optimize-restarts,svc-poly,gbdt-query-rmse,gmm-random-init-sample,gmm-sample,gp-normalize-y,gp-sample-y,gp-sample-y-normalize,gpc,gpc-multiclass,ivf-extend'.split(',')
def run(lane):
 with (out/(lane+'.log')).open('w') as log:
  code=subprocess.run(['.pixi/envs/test/bin/python','tools/verify_cpu_batch.py','--lanes',lane,'--out',str(out/lane),'--timeout','2400'],stdout=log,stderr=subprocess.STDOUT).returncode
 return lane,code
results={}
with ThreadPoolExecutor(max_workers=2) as pool:
 for future in as_completed([pool.submit(run,lane) for lane in lanes]):
  lane,code=future.result();results[lane]=code
  (out/'batch-exits.json').write_text(json.dumps(results,indent=2)+'\n')
  print(lane,code,flush=True)
raise SystemExit(int(any(results.values())))
PYBODY
