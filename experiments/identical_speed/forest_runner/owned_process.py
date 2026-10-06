"""Linux subreaper ownership includes workers that create separate sessions."""
import ctypes,os,signal,subprocess,time,pathlib
class CleanupUncertain(RuntimeError):pass

def snapshot():
 result={}
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   fields=(p/'stat').read_text().rsplit(')',1)[1].split()
   result[int(p.name)]=(int(fields[1]),fields[19],fields[0])
  except FileNotFoundError:pass
 return result

def descendants(table,root):
 owned=set();parents={root}
 while True:
  new={pid for pid,(ppid,_,_) in table.items() if ppid in parents}-owned
  if not new:return owned
  owned.update(new);parents=new

def run_owned(cmd,*,env,stdout,timeout,grace=5):
 # This dedicated controller must own no unrelated child work.
 libc=ctypes.CDLL(None,use_errno=True)
 if not hasattr(libc,'prctl') or libc.prctl(36,1,0,0,0)!=0:
  raise CleanupUncertain('Linux child subreaper could not be established')
 root=os.getpid();initial=snapshot()
 if descendants(initial,root):raise CleanupUncertain('Controller already has child processes')
 proc=subprocess.Popen(cmd,env=env,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True)
 tracked={};deadline=time.monotonic()+timeout;error=None;code=None
 def scan():
  table=snapshot()
  for pid in descendants(table,root):tracked[pid]=table[pid][1]
  return {pid:row for pid,row in table.items() if tracked.get(pid)==row[1] and row[2]!='Z'}
 try:
  while True:
   scan();code=proc.poll()
   if code is not None:break
   if time.monotonic()>=deadline:raise subprocess.TimeoutExpired(cmd,timeout)
   time.sleep(.1)
 except BaseException as exc:error=exc
 try:
  # Also clean descendants after a normal parent exit: detached children may persist.
  for sig in (signal.SIGTERM,signal.SIGKILL):
   until=time.monotonic()+grace
   while True:
    live=scan()
    if not live:break
    for pid,row in live.items():
     current=snapshot().get(pid)
     if current and current[1]==row[1]:
      try:os.kill(pid,sig)
      except ProcessLookupError:pass
    if time.monotonic()>=until:break
    time.sleep(.1)
  if scan():raise CleanupUncertain('Owned descendants remain after TERM/KILL')
  proc.wait(timeout=grace)
  while True:
   try:
    if os.waitpid(-1,os.WNOHANG)[0]==0:break
   except ChildProcessError:break
 except BaseException as exc:raise CleanupUncertain(str(exc)) from exc
 if error:raise error
 return code
