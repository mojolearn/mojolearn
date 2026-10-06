"""Serial native timing drivers with an explicit sealed opponent tail."""
import json,os,pathlib,re,signal,subprocess,time,traceback,hashlib
R=pathlib.Path('/root/overnight-nvidia');O=pathlib.Path('/root/campaign-results');P=O/'repairs';P.mkdir(exist_ok=True)
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def status(phase,**extra):save(O/'status.json',dict(phase=phase,pid=os.getpid(),time=time.time(),completed=len(results),queue_total=len(queue),**extra))
hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,compute_cap,memory.total','--format=csv,noheader'],text=True).strip()
results=json.loads((P/'results.json').read_text()) if (P/'results.json').exists() else {};queue=[]
base_env=dict(os.environ,LD_LIBRARY_PATH=str(R/'runtime'),LD_PRELOAD=str(R/'runtime/libgcc_s.so.1')+':'+str(R/'runtime/libstdc++.so.6'),MOJOLEARN_VENDOR='nvidia',MOJOLEARN_NUMERIC_MODE='identical',CUDA_VISIBLE_DEVICES='0')
while True:
 try:queue=json.loads((R/'repair-queue.json').read_text())
 except (json.JSONDecodeError,FileNotFoundError) as error:
  save(P/'queue-read-error.json',dict(time=time.time(),error=str(error)));time.sleep(2);continue
 pending=[x for x in queue if x['key'] not in results]
 if not pending:
  if (R/'CANDIDATES_SEALED').exists() and (R/'opponents-ready.json').exists() and not (P/'OPPONENTS_DONE').exists():
   (O/'DONE').unlink(missing_ok=True);status('GPU_OPPONENTS_RUNNING',hardware=hardware)
   with (P/'opponent-controller.log').open('a') as f:rc=subprocess.run(['python3',str(R/'opponent-tail.py')],stdout=f,stderr=subprocess.STDOUT).returncode
   if rc != 75:(P/'OPPONENTS_DONE').write_text(str(rc)+'\n')
   if rc == 75:continue
  if not (O/'DONE').exists():status('WAITING_FOR_NEXT_MEASUREMENTS',hardware=hardware);(O/'cmd.exit').write_text('0\n');(O/'DONE').touch()
  time.sleep(10);continue
 task=pending[0];(O/'DONE').unlink(missing_ok=True);(O/'cmd.exit').unlink(missing_ok=True);folder=P/task['key'];folder.mkdir(exist_ok=True);result=dict(task,hardware=hardware,machine=hardware,warmup_scope='same_process',scored_samples=1,warmups=1,started=time.time(),identity='inherited; no identity rerun')
 try:
  binary=R/'bin'/task['job'];assert hashlib.sha256(binary.read_bytes()).hexdigest()==task['binary_sha256'];binary.chmod(0o755)
  with (folder/'measurement.log').open('w') as f:
   process=subprocess.Popen([str(binary)],cwd=folder,env=dict(base_env,**task.get('environment',{})),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   status('MEASURING',current=task['key'],worker_pid=process.pid,hardware=hardware)
   try:rc=process.wait(timeout=task.get('timeout_seconds',1800))
   except subprocess.TimeoutExpired:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    rc=124
  text=(folder/'measurement.log').read_text(errors='replace');measure=[]
  for line in text.splitlines():
   if line.startswith(('MEASURE ','PAIRED ')):
    row=dict(re.findall(r'(\w+)=([^ ]+)',line))
    for k,v in list(row.items()):
     if v.isdigit():row[k]=int(v)
    if row.get('phase',1)==1:measure.append(row)
   elif line.startswith('ARM ') and '-score ' in line:
    tokens=line.split();measure.append(dict(phase=1,elapsed_ns=int(float(tokens[-1])*1e6)))
  result.update(returncode=rc,status='MEASURED' if rc==0 and measure else 'MEASUREMENT_FAILED',measurements=measure)
 except BaseException as e:result.update(status='MEASUREMENT_FAILED',returncode=1,error=repr(e));(folder/'error.log').write_text(traceback.format_exc())
 result['finished']=time.time();save(folder/'receipt.json',result);results[task['key']]=result;save(P/'results.json',results);status('CELL_FINISHED',current=task['key'],outcome=result['status'],hardware=hardware)
