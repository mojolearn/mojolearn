"""Measurement-first full RF/ET collection; failed cells remain explicit."""
import pathlib,json,os,sys,time,traceback
from owned_process import run_owned,CleanupUncertain
from admit import admit
R=pathlib.Path(sys.argv[1]);O=pathlib.Path(sys.argv[2]);D=json.loads((R/'packet.json').read_text());name=D.get('result_subdir','forest-ab');assert isinstance(name,str) and name not in ('','.','..') and pathlib.Path(name).name==name;C=O/name;C.mkdir(exist_ok=False);rc=1;clean=True;outcomes={}
def state(phase,**kw):
 p=C/'status.tmp';p.write_text(json.dumps({'phase':phase,'pid':os.getpid(),'time':time.time(),'source_sha':D['source_sha'],'harness_sha':D['harness_sha'],'outcomes':outcomes,'promotion':False,'qualification':'PENDING_CROSS_VENDOR_REFERENCE','measurement_first':True,**kw},indent=2));p.replace(C/'status.json')
try:
 assert admit(R)==D
 from settings import worker_environment,planned_cells
 env=worker_environment(D['vendor'],R/'runtime',D['data_environment']);wanted=planned_cells(D)
 os.chdir(R/'harness')
 for lane,datasets,profiles in [('rf',['taxi','istella'],['rf-k4','rf-k1','rf-k2','rf-k8']),('et',['istellareg','year'],['et-float','et-u16'])]:
  for dataset in datasets:
   for arm in profiles:
    if arm not in D['arms']:continue
    assert admit(R)==D
    cell=lane+'-'+dataset+'-'+arm
    if cell not in wanted:continue
    out=C/cell;out.mkdir();py=str(R/arm/'env/bin/python');manifest=R/arm/'manifest.json'
    family='identical/_mojolearn_'+('rf' if lane=='rf' else 'trees')+'.so'
    cmd=[py,str(R/'scored_worker.py'),'--artifact-manifest',str(manifest),'--required-binding',family,'--receipt',str(out/'worker-receipt.json'),'--harness',str(R/'harness'),'--','--lane',lane,'--dataset',dataset,'--ours-only','--save-scored-models',str(out/'scored')]
    (out/'invocation.json').write_text(json.dumps({'argv':cmd,'settings':{'warmups':1,'scored_samples':1,'rows':'full','opponents':False},'environment':{k:v for k,v in env.items() if k.startswith('MOJOLEARN_') or k in ['LD_LIBRARY_PATH','LD_PRELOAD','GBM_BENCH_DATA']}},indent=2))
    state('RUNNING',active_cell=cell)
    try:
     with (out/'worker.log').open('w') as log:code=run_owned(cmd,env=env,stdout=log,timeout=14520)
     outcomes[cell]={'worker_exit':code,'status':'FAILED'}
     if code==0:
      with (out/'audit.log').open('w') as log:audit=run_owned([py,str(R/'validate.py'),str(out),str(R/'harness'),lane,dataset],env=env,stdout=log,timeout=300)
      outcomes[cell].update(audit_exit=audit,status='MEASURED_STATE_CAPTURED' if audit==0 else 'FAILED_VALIDATION')
    except CleanupUncertain:raise
    except BaseException as exc:outcomes[cell]={'status':'FAILED','error':repr(exc)};(out/'exception.log').write_text(traceback.format_exc())
    state('RUNNING',last_cell=cell)
 expected=wanted
 assert set(outcomes)==expected and expected,'incomplete scheduled cell set'
 rc=0 if outcomes and all(v['status']=='MEASURED_STATE_CAPTURED' for v in outcomes.values()) else 1
 state('FULL_COLLECTION_CAPTURED' if rc==0 else 'FULL_COLLECTION_WITH_FAILURES',expected_cells=len(wanted))
except CleanupUncertain as exc:clean=False;state('CLEANUP_UNCERTAIN',error=repr(exc),child_cleanup_uncertain=True)
except BaseException as exc:(C/'error.log').write_text(traceback.format_exc());state('CONTROLLER_FAILED',error=repr(exc))
finally:
 if clean:(O/'cmd.exit').write_text(str(rc)+'\n');(O/'DONE').touch()
sys.exit(rc)
