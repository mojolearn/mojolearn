"""Recover exactly one owned worker; never overlap surviving measurement children."""
import fcntl,json,pathlib,subprocess,time
R=pathlib.Path('/root/overnight-nvidia');O=pathlib.Path('/root/campaign-results')
with (R/'worker-recovery.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 workers=[];children=[]
 for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
  try:argv=p.read_bytes().split(b'\0')
  except OSError:continue
  if b'/root/overnight-nvidia/repair-worker.py' in argv:workers.append(int(p.parent.name))
  elif any(a.startswith(b'/root/overnight-nvidia/bin/') or a in [b'/root/overnight-nvidia/opponent-tail.py',b'/root/overnight-nvidia/gpu-only-board.py'] for a in argv):children.append(int(p.parent.name))
 if workers:result=dict(status='ALIVE',pids=workers)
 elif children:result=dict(status='ORPHAN_CHILD_ACTIVE',pids=children)
 else:
  try:state=json.loads((O/'status.json').read_text())
  except (FileNotFoundError,json.JSONDecodeError):state={}
  setup=state.get('phase')=='OPPONENT_DEPENDENCY_STAGING' and pathlib.Path('/proc/'+str(state.get('pid'))+'/cmdline').exists()
  if setup:result=dict(status='DEPENDENCY_SETUP_ACTIVE',pid=state.get('pid'))
  else:
   # Refuse corrupt queues until a complete atomic publication arrives.
   queue=json.loads((R/'repair-queue.json').read_text())
   f=(O/'repair-controller.log').open('ab');p=subprocess.Popen(['python3',str(R/'repair-worker.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
   tmp=O/'repair-launch.recovery';tmp.write_text(json.dumps(dict(pid=p.pid,time=time.time())));tmp.replace(O/'repair-launch.json');result=dict(status='RESTARTED',pid=p.pid,queue_total=len(queue),time=time.time())
   with (O/'worker-recovery.jsonl').open('a') as record:record.write(json.dumps(result)+'\n')
 print(json.dumps(result))
