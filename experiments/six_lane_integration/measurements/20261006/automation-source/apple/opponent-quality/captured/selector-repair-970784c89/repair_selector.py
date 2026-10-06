"""Retry only three pre-fit selector failures, within the original CPU14 budget."""
from pathlib import Path
import fcntl,hashlib,importlib.util,json,os,subprocess,time,traceback
ROOT=Path('/Volumes/MojoPerfBuild20261005/six-lane-full-ab-20261006/apple/opponent-quality')
ORIGINAL=ROOT/'run-01';OUT=ROOT/'run-01-selector-repair'
SOURCE=ROOT.parent/'source-47301d12';RUNTIME=Path('/Users/ec2-user/mojolearn-full-f867b50e8')
SHA='47301d12b14859e81cadc9ab6a0cd4f728d0e206'
spec=importlib.util.spec_from_file_location('cpu14',SOURCE/'experiments/performance_ideas/apple_opponent_quality_20261006/run.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
OUT.mkdir(exist_ok=True)
assert not (OUT/'status.json').exists(),'Never reset a repair attempt'
state={'status':'WAITING','pid':os.getpid(),'source_sha':SHA,'total':3,'completed':0,'failed':0,'budget_limited':0,'budget_seconds':7200,'budget_extended':False,'cells':[]}
def read(p):return json.loads(p.read_text())
def save(**changes):
 state.update(changes,heartbeat_at=time.time());helper.atomic(OUT/'status.json',state)
try:
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()==SHA
 plan=read(ORIGINAL/'frozen-plan.json')['cells'][:3]
 original=read(ORIGINAL/'status.json');deadline=original['deadline_at']
 save(deadline_at=deadline,original_started_at=original['started_at'],phase='Wait for active CPU14 to terminate before failed-only repair')
 while read(ORIGINAL/'status.json')['status'] not in ('COMPLETE','COMPLETE_WITH_UNRESOLVED_CELLS','BLOCKED_FULL_INPUT_METADATA'):
  save();time.sleep(15)
 with (RUNTIME/'gpu.lock').open('a') as lock:
  while True:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:save();time.sleep(2)
  assert read(ORIGINAL/'status.json')['deadline_at']==deadline
  for name in ('tmp','cells'): (OUT/name).mkdir(exist_ok=True)
  driver=Path(__file__).with_name('selected_cpu_board.py')
  helper.atomic(OUT/'harness-provenance.json',{'source_sha':SHA,'driver':str(driver),'driver_sha256':hashlib.sha256(driver.read_bytes()).hexdigest(),'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reason':'Explicit sklearn existing-roster cells were removed by GPU-preference planning before any fit; owner authorized CPU opponents','original_deadline_at':deadline,'original_attempt':str(ORIGINAL),'same_data_settings_and_sample_policy':True})
  env={k:v for k,v in os.environ.items() if k not in helper.CAPS and not k.startswith('MOJOLEARN_')}
  env.update(TMPDIR=str(OUT/'tmp'),PYTHONDONTWRITEBYTECODE='1',DYLD_LIBRARY_PATH=str(RUNTIME/'runtime-libs'),GBM_BENCH_DATA='/Users/ec2-user/datasets/gbm-bench')
  for index,cell in enumerate(plan,1):
   old=read(ORIGINAL/'cells'/('cell-%02d'%index)/'result.json')
   assert old['status']=='FAILED' and old['returncode']==1 and old['race']==cell['race']
   path=OUT/'cells'/('cell-%02d'%index);path.mkdir()
   result={k:old[k] for k in ('race','arm','input_receipt','expected_fit_shape','expected_eval_shape')}
   result.update(status='NOT_RUN_BUDGET_EXHAUSTED',returncode=None,original_failure=str(ORIGINAL/'cells'/('cell-%02d'%index)/'result.json'))
   allowance=min(cell['seconds'],deadline-time.time());process=None;killed=[]
   if allowance>0:
    command=old['command'][:];command[2]=str(driver)
    for option,value in (('--out',str(path/'board')),('--opponent-store',str(path/'opponent-store.jsonl')),('--round-seconds',str(max(1,int(allowance))))):command[command.index(option)+1]=value
    started=time.time();save(status='RUNNING',active_race=cell['race'],active_arm=cell['arm'])
    try:
     with (path/'run.log').open('x') as log:process=subprocess.Popen(command,cwd=SOURCE,env=dict(env,REPAIR_RACE_ID=cell['race'],REPAIR_ARM=cell['arm']),stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
     while process.poll() is None:
      if time.time()>=min(deadline,started+allowance):killed=helper.stop_tree(process);break
      save(worker_pid=process.pid);time.sleep(2)
     result.update(command=command,returncode=process.returncode,status='BUDGET_LIMIT' if killed else 'FAILED',elapsed_seconds=time.time()-started,killed_owned_pids=killed)
     coverage=helper.actual_coverage(path/'board/board.json',cell);result.update({k:v for k,v in coverage.items() if k!='status'})
     if not killed and process.returncode==0:result['status']=coverage['status']
    except Exception as exc:
     if process is not None and process.poll() is None:killed=helper.stop_tree(process)
     result.update(status='FAILED_CONTROLLER',error=repr(exc),returncode=process.returncode if process else None,killed_owned_pids=killed)
   helper.atomic(path/'result.json',result)
   state['completed']+=result['status']=='MEASURED';state['failed']+=result['status'].startswith('FAILED');state['budget_limited']+=result['status'] in ('BUDGET_LIMIT','NOT_RUN_BUDGET_EXHAUSTED')
   state['cells'].append({k:result.get(k) for k in ('race','arm','status','returncode')});save(worker_pid=None)
 save(status='COMPLETE' if state['completed']==3 else 'COMPLETE_WITH_UNRESOLVED_CELLS',finished_at=time.time(),active_race=None,active_arm=None)
except BaseException as exc:
 save(status='FAILED_CONTROLLER',error=repr(exc),traceback=traceback.format_exc());raise
