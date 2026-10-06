"""One explicit vendor contract for real RF/ET fits; no inherited experiment flags."""
import os
from pathlib import Path

def worker_environment(vendor,runtime,data_environment,base=None):
 assert vendor in ('nvidia','amd')
 assert set(data_environment)=={'GBM_BENCH_DATA'}
 env={k:v for k,v in (os.environ if base is None else base).items()
      if not k.startswith(('MOJOLEARN_','MODULAR_','MOJO_'))
      and k not in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','LD_PRELOAD')}
 rt=Path(runtime);preload=[]
 for prefix in ('libgcc_s.so.','libstdc++.so.'):
  matches=list(rt.glob(prefix+'*'))
  if len(matches)!=1:raise ValueError('one pinned runtime required: '+prefix)
  preload.append(str(matches[0]))
 actual='cuda' if vendor=='nvidia' else 'hip'
 # forest_speed_arm requires EXPECTED_VENDOR before it constructs ours. The
 # initial AMD launch omitted it and refused three cells before any fit.
 env.update(MOJOLEARN_VENDOR=actual,MOJOLEARN_SPEED_EXPECTED_VENDOR=actual,
            MOJOLEARN_NUMERIC_MODE='identical',MOJOLEARN_SPEED_ROUNDS='1',
            MOJOLEARN_SPEED_SIZE='shipped',MOJOLEARN_SPEED_BUDGET_S='7200',
            MOJOLEARN_SPEED_DEADLINE_S='14400',LD_LIBRARY_PATH=str(rt),
            LD_PRELOAD=':'.join(preload),PYTHONUNBUFFERED='1')
 env.update(data_environment)
 if vendor=='nvidia':env['MOJOLEARN_CUDA_PATH']='native'
 return env
