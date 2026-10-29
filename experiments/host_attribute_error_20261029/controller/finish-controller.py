"""Finite owner-requested completion controller; not a watcher service."""
from pathlib import Path
import datetime,json,os,signal,subprocess,sys,time,traceback
r=Path(__file__).parent
state={'schema':'mojolearn.host-attribute-finite-controller.v1','pid':os.getpid(),'source_commit':'543875dbb0b4b334153dbe9b61b3d7a60c177b2c','compile_pid':44266,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'waiting-existing-compile','steps':[],'scope':'Finish existing compile, then11 focused tests and9fixture18part CPU crosscheck; no broad campaign or new Mamba work'}
def save():
 p=r/'finish-state.json.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(r/'finish-state.json')
def run(name,script,timeout):
 state['status']=name;save()
 with (r/(name+'-controller.log')).open('w') as log:
  process=subprocess.Popen([sys.executable,str(r/script)],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=process.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   rc=124
 state['steps'].append({'step':name,'exit_code':rc});save()
 if rc:raise RuntimeError(name+' exit '+str(rc))
save();failure=None
try:
 deadline=time.monotonic()+45*60
 while not (r/'build.exit').exists():
  if time.monotonic()>deadline:raise RuntimeError('existing compile completion deadline exceeded; no duplicate build launched')
  try:os.kill(44266,0)
  except ProcessLookupError:
   time.sleep(2)
   if not (r/'build.exit').exists():raise RuntimeError('existing compile owner exited without final status; no duplicate build launched')
  time.sleep(5)
 result=json.loads((r/'build/result.json').read_text())
 owners=[x for x in result.get('results',[]) if x.get('stem')=='_mojolearn_dev_runtime_classical_host']
 assert len(owners)==1 and owners[0]['exit_code']==0, 'runtime_classical compile did not pass'
 initial=json.loads((r/'initial-compile/build/result.json').read_text());owners=[x for x in initial['results'] if x.get('stem')=='_mojolearn_dev_tokenizer_host'];assert len(owners)==1 and owners[0]['exit_code']==0
 state['steps'].append({'step':'two-native-compiles','exit_code':0,'controller_exit':int((r/'build.exit').read_text()),'note':'partial-set controller exit3 is not full-release activation'});save()
 run('stage','stage.py',300)
 run('focused','run-focused.py',600)
 run('crosscheck','run-crosscheck.py',1200)
 run('context','context-fragment.py',120)
except BaseException as error:
 failure=repr(error);state['failure']=failure;traceback.print_exc()
finally:
 try:run('retention','retain.py',600)
 except BaseException as error:state['retention_failure']=repr(error);traceback.print_exc();failure=failure or repr(error)
 state['status']='failed' if failure else 'complete';state['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
 (r/'finish.exit').write_text(('1' if failure else '0')+'\n')
print(json.dumps({'status':state['status'],'steps':state['steps'],'failure':state.get('failure')}))
