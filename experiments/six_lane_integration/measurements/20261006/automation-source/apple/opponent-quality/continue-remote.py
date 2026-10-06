"""Durably wait for the two reserved Apple queues, then launch CPU14 once."""
from pathlib import Path
import fcntl,json,os,runpy,time,traceback
BASE=Path('/Volumes/MojoPerfBuild20261005/six-lane-full-ab-20261006/apple')
ROOT=BASE/'opponent-quality'
STATUS=ROOT/'continuation-status.json'
state={'status':'WAITING','pid':os.getpid(),'source_sha':'47301d12b14859e81cadc9ab6a0cd4f728d0e206','budget_seconds':7200,'budget_started':False}
def save(**updates):
 state.update(updates,heartbeat_at=time.time())
 p=STATUS.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2)+'\n');p.replace(STATUS)
def read(path):
 try:return json.loads(path.read_text())
 except FileNotFoundError:return {'status':'MISSING'}
with (ROOT/'continuation.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  while True:
   predecessors={n:read(BASE/n/'runs/owner-status.json') for n in ('gmm-istella-full','pls-qn-full')}
   save(status='WAITING',phase='Waiting for reserved own Apple queues to terminate',predecessors=predecessors)
   if all(v.get('status') in ('EXITED','FAILED_CONTROLLER') for v in predecessors.values()):break
   time.sleep(30)
  # The controller owns machine-lock acquisition and starts its unchanged
  # two-hour budget only after acquisition; waiting above cannot spend it.
  if not (ROOT/'controller.pid').exists():
   save(status='LAUNCHING',phase='Reserved queues terminal; launching authorized CPU14 once')
   runpy.run_path(str(ROOT/'launch.py'),run_name='__main__')
  pid=int((ROOT/'controller.pid').read_text())
  while True:
   result=read(ROOT/'run-01/status.json')
   terminal=result.get('status') in ('COMPLETE','COMPLETE_WITH_UNRESOLVED_CELLS','FAILED_CONTROLLER','FAILED_INPUT_PROVENANCE','FAILED_SOURCE_FREEZE','BLOCKED_FULL_INPUT_METADATA')
   save(status=result.get('status') if terminal else 'RUNNING',phase='Controller terminal' if terminal else 'Monitoring existing CPU14 controller',controller_pid=pid,controller_status=result,budget_started=result.get('budget_started',bool(result.get('started_at'))))
   if terminal:break
   try:os.kill(pid,0)
   except ProcessLookupError:raise RuntimeError('CPU14 controller exited without recognized terminal status; original evidence retained')
   time.sleep(30)
 except BaseException as exc:
  save(status='FAILED_CONTROLLER',error=repr(exc),traceback=traceback.format_exc())
  raise
