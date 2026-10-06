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

def planned_cells(packet):
 groups=[('rf',['taxi','istella'],['rf-k4','rf-k1','rf-k2','rf-k8']),('et',['istellareg','year'],['et-float','et-u16'])]
 universe={family+'-'+dataset+'-'+profile for family,datasets,profiles in groups for dataset in datasets for profile in profiles if profile in packet['arms']}
 selected=packet.get('selected_cells')
 if selected is None:return universe
 assert selected and all(row['vendor']==packet['vendor'] for row in selected)
 wanted={row['family']+'-'+row['dataset']+'-'+row['profile'] for row in selected}
 assert len(wanted)==len(selected) and wanted<=universe
 for row in selected:
  baseline='rf-k4' if row['family']=='rf' else 'et-float'
  assert row['family']+'-'+row['dataset']+'-'+baseline in wanted,'selected candidate lacks baseline'
 return wanted
