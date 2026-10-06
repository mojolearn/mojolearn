import concurrent.futures,fcntl,hashlib,json,os,pathlib,subprocess,threading,time,traceback
B=pathlib.Path('/root/measurement-builder');R=B/'nvidia-more-source';O=B/'artifacts/nvidia-more';O.mkdir(parents=True,exist_ok=True);claim=B/'claims/nvidia-more';stop=False
plan=json.loads((B/'more-build-plan.json').read_text())
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def heartbeat():
 while not stop:claim.touch();time.sleep(30)
threading.Thread(target=heartbeat,daemon=True).start()
try:
 R.mkdir(exist_ok=True);subprocess.run(['tar','xzf',str(B/'more-source.tgz'),'-C',str(R)],check=True)
 while True:
  p=B/'artifacts/nvidia-family-repairs/status.json'
  if p.exists() and json.loads(p.read_text()).get('phase') in ['BUILDS_FINISHED','BUILD_CONTROLLER_FAILED']:break
  save(O/'status.json',dict(phase='WAITING_NVIDIA_COMPILE_SLOT',expected=len(plan),time=time.time()));time.sleep(15)
 envroot=B/'source/.pixi/envs/default';env=dict(os.environ,MODULAR_HOME=str(envroot/'share/max'),PATH=str(envroot/'bin')+':'+os.environ['PATH']);rows=[]
 for task in plan:
  name=task['job'];binary=O/name
  with (B/'nvidia-compile.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   save(O/'status.json',dict(phase='BUILDING_AFFECTED_MEASUREMENT_DRIVER',job=name,completed=len(rows),expected=len(plan),time=time.time()))
   cmd=[str(envroot/'bin/mojo'),'build','-j1','-I',str(R),'-D','MOJOLEARN_NUMERIC_IDENTICAL=1','-D','MOJOLEARN_COLUMN_'+task['vendor'].upper()+'=1','--target-accelerator',task['accelerator']]
   for d in task['defines']:cmd+=['-D',d]
   cmd += [str(R/task['source']),'-o',str(binary)]
   with (O/(name+'.log')).open('w') as f:
    try:rc=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800).returncode
    except subprocess.TimeoutExpired:rc=124
   row=dict(task,returncode=rc,build_argv=cmd)
   if rc==0:row['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
   rows.append(row);save(O/(name+'.json'),row);save(O/'builds.json',dict(builds=rows))
 save(O/'status.json',dict(phase='BUILDS_FINISHED',completed=len(rows),expected=len(plan),failed=sum(x['returncode']!=0 for x in rows),time=time.time()))
except BaseException as e:
 (O/'error.log').write_text(traceback.format_exc());save(O/'status.json',dict(phase='BUILD_CONTROLLER_FAILED',error=repr(e),time=time.time()))
finally:stop=True;claim.unlink(missing_ok=True)
