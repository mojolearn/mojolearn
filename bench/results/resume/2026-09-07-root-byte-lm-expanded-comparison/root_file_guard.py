import os, sys, json, time, subprocess, signal
from pathlib import Path
out=Path(sys.argv[1]); cmd=sys.argv[2:]
assert cmd and not out.with_suffix('.log').exists()
env=dict(os.environ)
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','MAX_JOBS','CMAKE_BUILD_PARALLEL_LEVEL'): env[k]='2'
env['PYTHONPATH']='/Users/andrewhendel/CascadeProjects/mojolearn/python:/Users/andrewhendel/CascadeProjects/mojolearn/tools'
started=time.monotonic();peak=0;reason=None
with out.with_suffix('.log').open('x') as log:
 p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  while p.poll() is None:
   if time.monotonic()-started>60: reason='60-second deadline';break
   rows=subprocess.run(['ps','-axo','pgid=,rss='],capture_output=True,text=True,check=True,timeout=2).stdout.splitlines()
   rss=sum(int(parts[1]) for row in rows if len(parts:=row.split())==2 and int(parts[0])==p.pid)
   peak=max(peak,rss)
   if rss>1048576: reason='1GiB RSS cap';break
   time.sleep(.1)
 except BaseException as e:
  reason=repr(e)
 finally:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGKILL)
  rc=p.wait(timeout=5)
record=dict(command=cmd,returncode=rc,reason=reason,peak_rss_kib=peak,elapsed_seconds=time.monotonic()-started,thread_limit=2,cpu_affinity=None,scope='Root bounded file checks, mocks or document build; no native GPU compiler/model; sampled RSS bound, no hard Darwin CPU affinity')
out.with_suffix('.guard.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
sys.exit(1 if reason else rc)
