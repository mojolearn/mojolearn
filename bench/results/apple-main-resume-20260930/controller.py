import subprocess,os,json,time,fcntl,shlex
from pathlib import Path
H=Path.home();B=H/'mojolearn-evidence/cloud-retirement-20260930';W=H/'mojolearn-wt/board-resume-20260930';os.umask(0o077)
f=open(B/'resume-controller.lock','w')
try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit()
def run(args,timeout=120):
 p=subprocess.run(args,cwd=W,capture_output=True,text=True,timeout=timeout,env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'})
 if p.returncode:raise RuntimeError(p.stderr[-600:] or p.stdout[-600:])
 return p.stdout
code='''import json,pathlib
h=pathlib.Path.home();root=h/'board-resume';cfg=json.loads((root/'resume-config.json').read_text());d=json.loads((root/'board/board.json').read_text()) if (root/'board/board.json').exists() else {'races':{}};exc=json.loads((root/'exceptions.json').read_text()) if (root/'exceptions.json').exists() else {}
q=h/'gpu-queue'
if q.exists():active=[str(p) for p in q.glob('[0-9]*/status') if p.read_text().strip() in ['queued','starting','running']]
else:active=[str(p) for s in ['queue','working'] for p in (h/'mojolearn-evidence/apple-steward'/s).glob('[0-9]*.json')]
print(json.dumps({'pending':len(set(cfg['plan'])-d['races'].keys()-exc.keys()),'missing':len(set(cfg['plan'])-d['races'].keys()),'blocked':(root/'blocked.json').exists(),'active':active,'data_ready':(root/'DATA_READY').exists()}))'''
p=B/'resume-controller-state.json';state=json.loads(p.read_text()) if p.exists() else {}
targets=[('amd','root@162.243.212.105','22'),('m3ultra-b','ec2-user@54.157.1.251','22'),('m2pro','ec2-user@54.237.205.45','22')]
np=H/'mojolearn-evidence/nvidia_central/lanes/board-resume/pod'
if np.exists():
 sp=H/'mojolearn-evidence/devpods'/np.read_text().strip()/'state.env'
 if sp.exists():
  vals={k:(shlex.split(v)[0] if shlex.split(v) else '') for k,v in (line.split('=',1) for line in sp.read_text().splitlines())}
  parts=shlex.split(vals['SSH_TARGET']);port=parts[parts.index('-p')+1] if '-p' in parts else '22';targets.insert(0,('nvidia',parts[-1],port))
override_file=B/'target-overrides.json'
overrides=json.loads(override_file.read_text()) if override_file.exists() else {}
for name,host,port in targets:
 try:
  override=overrides.get(name,{})
  remote_root=override.get('root','/Users/ec2-user/board-resume' if name.startswith('m') else '/root/board-resume')
  target_code=code.replace("root=h/'board-resume'",'root=pathlib.Path('+repr(remote_root)+')')
  ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-i',str(H/'.ssh'/('mambik-l8.pem' if name.startswith('m') else 'id_ed25519')),'-p',port,host]
  s=json.loads(run(ssh+['python3 -c '+shlex.quote(target_code)],60));old=state.get(name,{});s['checked']=time.time();s['last_backup']=old.get('last_backup',0);s['last_extend']=old.get('last_extend',0)
  if name=='amd' and time.time()-s['last_extend']>900:
   run(['bash','tools/do_amd_steward.sh','extend','120'],180);s['last_extend']=time.time()
  if name=='nvidia':run(['bash','tools/nvidia_central.sh','sh','board-resume','true'],60)
  if time.time()-s['last_backup']>600:
   dest=B/'resumed-results';dest.mkdir(exist_ok=True);tmp=dest/(name+'.partial')
   with tmp.open('wb') as out:
    r=subprocess.run(ssh+['tar --exclude=cache --exclude=.pixi -czf - -C '+remote_root+' .'],stdout=out,stderr=subprocess.PIPE,timeout=300)
   if r.returncode==0:tmp.replace(dest/(name+'.tar.gz'));s['last_backup']=time.time()
  if s['pending'] and not s['active'] and not s['blocked'] and s['data_ready']:
   if name=='nvidia':result=run(['bash','tools/nvidia_central.sh','submit','board-resume','--cap','240','/root/board-resume/run.sh'])
   else:
    target='do-amd' if name=='amd' else name
    command='test -f /root/board-resume/DATA_READY && /usr/bin/python3 tools/bench_board_resume.py --root /root/board-resume' if name=='amd' else 'test -f \"$HOME/board-resume/DATA_READY\" && \"$HOME/mojolearn/.pixi/envs/default/bin/python3.13\" tools/bench_board_resume.py --root \"$HOME/board-resume\"'
    command=override.get('command',command)
    result=run(['python3','tools/apple_steward.py','submit','--kind','speed','--lane','board-resume','--commit',override.get('commit','a19d159d7'),'--target',target,'--cmd',command])
   s['submitted']=result.strip()
  state[name]=s;print(time.strftime('%FT%TZ',time.gmtime()),name,json.dumps(s),flush=True)
 except Exception as e:print(name,'ERROR',str(e),flush=True)
p.write_text(json.dumps(state,indent=2)+'\n')
