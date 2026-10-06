import fcntl,hashlib,json,os,pathlib,subprocess,threading,time,traceback
R=pathlib.Path('/root/measurement-builder/source');O=pathlib.Path('/root/measurement-builder/artifacts/nvidia-family-repairs');O.mkdir(exist_ok=True,parents=True);claim=pathlib.Path('/root/measurement-builder/claims/nvidia-family-repairs');claim.parent.mkdir(exist_ok=True);stop=False
SOURCE='f32549d28'
def heartbeat():
 while not stop:claim.touch();time.sleep(30)
threading.Thread(target=heartbeat,daemon=True).start()
def save(name,data):
 p=O/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
try:
 with open('/root/measurement-builder/compile.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  while not pathlib.Path('/root/.pixi/bin/pixi').exists():time.sleep(5)
  with (O/'environment.log').open('w') as f:
   q=subprocess.run(['/root/.pixi/bin/pixi','install','--locked'],cwd=R,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
  if q.returncode:raise RuntimeError('locked environment restore failed')
  env=dict(os.environ,PATH='/root/.pixi/bin:'+str(R/'.pixi/envs/default/bin')+':'+os.environ['PATH'],MODULAR_HOME=str(R/'.pixi/envs/default/share/max'))
  plan=[('paired','experiments/performance_ideas/paired_gemm_time.mojo',[]),('I15-baseline','experiments/performance_ideas/I15/time.mojo',['MOJOLEARN_KNN_SELECT_TRIAL=1','MOJOLEARN_IDN_KNN_CERTIFIED_REACH=1','MOJOLEARN_KNN_CERTIFIED_MMA_OFF=1']),('I15-candidate','experiments/performance_ideas/I15/time.mojo',['MOJOLEARN_KNN_SELECT_TRIAL=1','MOJOLEARN_IDN_KNN_CERTIFIED_REACH=1']),('I20-baseline','experiments/performance_ideas/I20/time.mojo',[]),('I20-candidate','experiments/performance_ideas/I20/time.mojo',['MOJOLEARN_IDN_KDE_PARTIAL_POOL=1']),('I24-baseline','experiments/performance_ideas/I24/time.mojo',[]),('I24-candidate','experiments/performance_ideas/I24/time.mojo',['MOJOLEARN_METRICS_JOINT_COUNTS=1'])]
  rows=[]
  for vendor,arch in [('nvidia','sm_89'),('amd','gfx942')]:
   for name,source,defines in plan:
    job=vendor+'-'+name;binary=O/job;cmd=[str(R/'.pixi/envs/default/bin/mojo'),'build','-j1','-I',str(R),'-D','MOJOLEARN_NUMERIC_IDENTICAL=1','-D','MOJOLEARN_COLUMN_'+vendor.upper()+'=1','--target-accelerator',arch]
    for d in defines:cmd+=['-D',d]
    cmd+=[str(R/source),'-o',str(binary)];save('status.json',dict(phase='BUILDING_AFFECTED_MEASUREMENT_DRIVER',job=job,completed=len(rows),expected=len(plan)*2,time=time.time()))
    start=time.time()
    with (O/(job+'.log')).open('w') as f:
     try:rc=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800).returncode
     except subprocess.TimeoutExpired:rc=124
    row=dict(job=job,name=name,vendor=vendor,accelerator=arch,source_sha=SOURCE,source=source,defines=defines,build_argv=cmd,returncode=rc,elapsed=time.time()-start)
    if rc==0:row['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    rows.append(row);save('builds.json',dict(source_sha=SOURCE,builds=rows));save(job+'.json',row)
  save('status.json',dict(phase='BUILDS_FINISHED',completed=len(rows),failed=sum(x['returncode']!=0 for x in rows),time=time.time()))
except BaseException as e:
 (O/'controller-error.log').write_text(traceback.format_exc());save('status.json',dict(phase='BUILD_CONTROLLER_FAILED',error=repr(e),time=time.time()))
finally:
 stop=True;claim.unlink(missing_ok=True)
