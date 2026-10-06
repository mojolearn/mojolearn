"""Serial retained-artifact measurements; mutable queue accepts repaired drivers."""
import argparse,hashlib,json,os,pathlib,re,signal,subprocess,time,traceback
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=pathlib.Path,default=pathlib.Path('/root/overnight-nvidia'));parser.add_argument('--output',type=pathlib.Path,default=pathlib.Path('/root/campaign-results'));args=parser.parse_args();R=args.root;O=args.output;O.mkdir(exist_ok=True)
def save(path,value):
 p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(path)
def status(phase,**kw):
 save(O/'status.json',dict(phase=phase,pid=os.getpid(),time=time.time(),completed=len(done),queue_total=len(queue),**kw))
hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,compute_cap,memory.total','--format=csv,noheader'],text=True).strip()
done={};queue=[]
if (O/'results.json').exists():done=json.loads((O/'results.json').read_text())
env=dict(os.environ,LD_LIBRARY_PATH=str(R/'runtime'),LD_PRELOAD=str(R/'runtime/libgcc_s.so.1')+':'+str(R/'runtime/libstdc++.so.6'),MOJOLEARN_VENDOR='nvidia',MOJOLEARN_NUMERIC_MODE='identical',CUDA_VISIBLE_DEVICES='0')
while True:
 queue=json.loads((R/'queue.json').read_text())
 pending=[q for q in queue if q['key'] not in done]
 if not pending:
  if not (O/'DONE').exists():
   status('WAITING_FOR_NEXT_MEASUREMENTS',hardware=hardware,idle_policy='30 minutes after verified capture; pending driver repairs may append queue')
   (O/'cmd.exit').write_text('0\n');(O/'DONE').touch()
  time.sleep(15);continue
 (O/'DONE').unlink(missing_ok=True);(O/'cmd.exit').unlink(missing_ok=True)
 q=pending[0];d=O/q['key'];d.mkdir(exist_ok=True)
 result=dict(q,hardware=hardware,started=time.time(),identity='inherited accepted prior evidence; no separate identity validation',compilation='reused existing green artifact',full_operation_qualification=False,promotion=False,one_sample=True,warmup_scope='separate_process',steady_state_claim=False,baseline_machine=hardware,candidate_machine=hardware,baseline_source_sha=q['source_sha'],candidate_source_sha=q['source_sha'],runs=[])
 try:
  binary=R/'bin'/q['job'];binary.chmod(0o755);result['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
  assert result['binary_sha256']==q['binary_sha256'],'artifact bytes changed'
  for phase in ['warmup','scored']:
   status('MEASURING',current=q['key'],sample=phase,hardware=hardware)
   log=d/(phase+'.log');started=time.time();rc=127
   with log.open('w') as stream:
    process=subprocess.Popen([str(binary)],cwd=d,env=dict(env,**q['environment']),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    status('MEASURING',current=q['key'],sample=phase,worker_pid=process.pid,hardware=hardware)
    try:rc=process.wait(timeout=q.get('timeout_seconds',1800))
    except subprocess.TimeoutExpired:
     os.killpg(process.pid,signal.SIGTERM)
     try:process.wait(timeout=20)
     except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
     rc=124
   text=log.read_text(errors='replace');measurements=[]
   for line in text.splitlines():
    if re.search(r'(?:completion_ns|elapsed_ns|timing_ns)=',line):
     row=dict(re.findall(r'([A-Za-z0-9_]+)=([^ ]+)',line))
     for k,v in list(row.items()):
      if v.isdigit():row[k]=int(v)
     measurements.append(row)
   result['runs'].append(dict(phase=phase,returncode=rc,wall_seconds=time.time()-started,measurement_count=len(measurements),measurements=measurements if phase=='scored' else [],excluded=phase=='warmup',log=str(log)))
   if rc:break
  scored=[x for x in result['runs'] if x['phase']=='scored']
  result['status']='MEASURED_COMPONENT' if scored and scored[0]['returncode']==0 and scored[0]['measurements'] else 'MEASUREMENT_FAILED'
  if q.get('unpaired'):result['qualification']='candidate-only component timing; baseline driver repair pending'
  else:result['qualification']='component timing only; full public caller measurement pending'
 except BaseException as e:
  result.update(status='MEASUREMENT_FAILED',error=repr(e));(d/'error.log').write_text(traceback.format_exc())
 result['finished']=time.time();save(d/'receipt.json',result);done[q['key']]=result;save(O/'results.json',done)
 status('CELL_FINISHED',current=q['key'],outcome=result['status'],hardware=hardware)
