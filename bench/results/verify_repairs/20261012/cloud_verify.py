"""Task-owned verification boxes. Never destroy until artifacts are captured."""
import argparse, hashlib, json, os, pathlib, shlex, subprocess, tempfile, time
ROOT=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('provider',choices=['runpod','do']);p.add_argument('action',choices=['start','status','fetch','delete']);a=p.parse_args()
out=ROOT/a.provider;out.mkdir(exist_ok=True)
statefile=out/'state.json'
state=json.loads(statefile.read_text()) if statefile.exists() else {}
def save(): statefile.write_text(json.dumps(state,indent=2)+'\n')
def api(method,path,data=None):
 key=pathlib.Path(os.environ.get('MOJOLEARN_RUNPOD_KEY_FILE',str(pathlib.Path.home()/'.mojolearn_runpod_key'))) if a.provider=='runpod' else pathlib.Path(os.environ.get('MOJOLEARN_DO_TOKEN_FILE',str(pathlib.Path.home()/'.mojolearn_do_token')))
 base='https://rest.runpod.io/v1' if a.provider=='runpod' else 'https://api.digitalocean.com/v2'
 with tempfile.TemporaryDirectory(prefix='ml-verify-api-') as d:
  d=pathlib.Path(d);c=d/'curlrc';c.write_text('header = "Authorization: Bearer '+key.read_text().strip()+'"\nsilent\nshow-error\n');c.chmod(0o600)
  cmd=['curl','-K',str(c),'--max-time','60','-X',method,'-o',str(d/'body'),'-w','%{http_code}',base+path]
  if data is not None:
   (d/'request').write_text(json.dumps(data));cmd+=['-H','Content-Type: application/json','--data-binary','@'+str(d/'request')]
  r=subprocess.run(cmd,capture_output=True,text=True);code=int(r.stdout or 0)
  b=(d/'body').read_text() if (d/'body').exists() else ''
  try:body=json.loads(b)
  except ValueError:body={'unparsed':b[:100]}
  return code,body
endpoint='/pods' if a.provider=='runpod' else '/droplets'
def ssh(cmd,input=None,timeout=60,check=True):
 return subprocess.run(['ssh',*state['ssh'],cmd],input=input,text=True,capture_output=True,timeout=timeout,check=check)
def inspect():
 code,b=api('GET',endpoint+'/'+str(state['id']))
 return code, b if a.provider=='runpod' else b.get('droplet',b)
if a.action=='start':
 if state:raise SystemExit('Existing state; refusing duplicate rental')
 name='mojolearn-verify-neural-'+a.provider+'-'+time.strftime('%Y%m%d%H%M%S',time.gmtime())
 state.update(name=name,provider=a.provider,created_at=time.time(),phase='creating');save()
 if a.provider=='runpod':
  req=dict(name=name,imageName='runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04',gpuTypeIds=['NVIDIA GeForce RTX 4090'],gpuCount=1,cloudType='SECURE',containerDiskInGb=80,volumeInGb=0,ports=['22/tcp'],supportPublicIp=True,interruptible=False,allowedCudaVersions=['13.0'])
 else:
  code,b=api('GET','/droplets?per_page=200')
  if code!=200 or any(x.get('size_slug','').startswith('gpu-') for x in b['droplets']):raise SystemExit('DO inventory unavailable or another GPU already exists')
  req=dict(name=name,region='tor1',size='gpu-mi325x1-256gb',image=188571990,ssh_keys=['df:f7:6b:0c:56:da:48:a5:6f:6d:ae:44:af:de:f3:0b'],tags=['mojolearn-verify-neural'])
 (out/'create-request.json').write_text(json.dumps(req,indent=2))
 code,b=api('POST',endpoint,req)
 obj=b if a.provider=='runpod' else b.get('droplet',{})
 if not obj.get('id'):
  state.update(phase='create_failed',http=code,error=b.get('message',b.get('error','unparsed response')));save();print(json.dumps(state));raise SystemExit(1)
 state.update(id=obj['id'],phase='waiting_ssh',cost_per_hour=obj.get('costPerHr') or obj.get('adjustedCostPerHr') or obj.get('size',{}).get('price_hourly'));save();print(json.dumps(state),flush=True)
 deadline=time.time()+900
 while time.time()<deadline:
  code,obj=inspect()
  if a.provider=='runpod':ip=obj.get('publicIp');port=(obj.get('portMappings') or {}).get('22')
  else:
   ips=[n['ip_address'] for n in obj.get('networks',{}).get('v4',[]) if n['type']=='public'];ip=ips[0] if ips else None;port=22
  if ip and port:
   state['ssh']=['-p',str(port),'-i',str(pathlib.Path.home()/'.ssh/id_ed25519'),'-o','StrictHostKeyChecking=accept-new','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=4','root@'+ip];save()
   try:
    if ssh('true',timeout=25,check=False).returncode==0:break
   except subprocess.TimeoutExpired:pass
  time.sleep(10)
 else:raise SystemExit('SSH not ready; instance retained and recorded for recovery')
 job=(ROOT/'cloud-job.sh').read_text()
 ssh('mkdir -p /root/ml-verify/artifacts && cat > /root/ml-verify/job.sh',input=job)
 r=ssh('nohup bash /root/ml-verify/job.sh > /root/ml-verify/artifacts/job.log 2>&1 < /dev/null & echo $!')
 state.update(phase='running',job_pid=int(r.stdout.strip()));save();print(json.dumps(state),flush=True)
elif a.action=='status':
 print('provider',a.provider,'id',state.get('id'),'phase',state.get('phase'))
 if state.get('ssh'):
  r=ssh("python3 - <<'CODE'\nimport pathlib,json\np=pathlib.Path('/root/ml-verify/artifacts')\nfor f in sorted(p.glob('*.exit')):print(f.name,f.read_text().strip())\nfor f in sorted(p.glob('*.progress.json')):\n d=json.loads(f.read_text()); print(f.name,d.get('execution'),d.get('elapsed_s'))\nCODE",check=False);print(r.stdout);print('ssh_exit',r.returncode)
elif a.action=='fetch':
 r=ssh("cd /root/ml-verify && tar czf capture.tar.gz artifacts && sha256sum capture.tar.gz",timeout=300)
 wanted=r.stdout.split()[0]
 with (out/'capture.tar.gz').open('wb') as f:
  subprocess.run(['ssh',*state['ssh'],'cat /root/ml-verify/capture.tar.gz'],stdout=f,check=True,timeout=1800)
 h=hashlib.file_digest((out/'capture.tar.gz').open('rb'),'sha256').hexdigest()
 if h!=wanted:raise SystemExit('capture hash mismatch; box retained')
 subprocess.run(['tar','xzf',str(out/'capture.tar.gz'),'-C',str(out)],check=True)
 state.update(capture_sha256=h,captured_at=time.time(),phase='captured');save();print('CAPTURED',a.provider,h)
elif a.action=='delete':
 if state.get('phase')!='captured':raise SystemExit('No verified artifact capture; refusing teardown')
 if not (out/'artifacts/job.exit').exists():raise SystemExit('Job not completed; refusing teardown')
 code,obj=inspect()
 if code==200 and obj.get('name')!=state['name']:raise SystemExit('Ownership mismatch')
 code,b=api('DELETE',endpoint+'/'+str(state['id']))
 for _ in range(20):
  code,obj=inspect()
  if code==404 or (a.provider=='runpod' and obj.get('desiredStatus')=='TERMINATED'):
   code2,ls=api('GET',endpoint+'?per_page=200' if a.provider=='do' else endpoint)
   items=ls.get('droplets',[]) if a.provider=='do' else (ls if isinstance(ls,list) else ls.get('items',ls.get('pods',[])))
   if code2==200 and all(str(x['id'])!=str(state['id']) for x in items):
    state.update(phase='terminated',terminated_at=time.time());save();print('TERMINATION_VERIFIED',a.provider,state['id']);break
  time.sleep(5)
 else:raise SystemExit('Deletion not verified')
