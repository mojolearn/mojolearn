import json,pathlib,subprocess,sys,time,os,signal,traceback
R=pathlib.Path('/root/overnight-nvidia');O=pathlib.Path('/root/campaign-results');T=O/'opponent-setup';T.mkdir(exist_ok=True);results=[]
try:
 while True:
  s=json.loads((O/'status.json').read_text())
  if s['phase']=='WAITING_FOR_NEXT_MEASUREMENTS':
   pid=json.loads((O/'repair-launch.json').read_text())['pid'];cmd=pathlib.Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(b'\0')
   if b'/root/overnight-nvidia/repair-worker.py' not in cmd:raise RuntimeError('repair worker ownership mismatch')
   os.kill(pid,signal.SIGSTOP)
   latest=json.loads((O/'status.json').read_text())
   if latest['phase']=='WAITING_FOR_NEXT_MEASUREMENTS':os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);break
   os.kill(pid,signal.SIGCONT)
  time.sleep(5)
 (O/'DONE').unlink(missing_ok=True);(O/'status.json').write_text(json.dumps(dict(phase='OPPONENT_DEPENDENCY_STAGING',pid=os.getpid(),time=time.time())))
 subprocess.run(['python3','-m','venv','/root/opponent-venv'],check=True)
 sys.path.insert(0,'/root/opponent-harness/tools');import bench_board as b
 pip=['/root/opponent-venv/bin/python','-m','pip','install','--no-input','--disable-pip-version-check']
 groups=[pip+['--extra-index-url','https://download.pytorch.org/whl/cu129','torch==2.13.0+cu129']]
 for idx,reqs in b.opponent_requirements('nvidia',b.opponent_pins()):groups.append(pip+['--no-index','--find-links','/root/opponent-wheels/trees-linux-x86_64-cp311','--find-links','/root/opponent-wheels/rapids-linux-x86_64-cp311']+reqs)
 groups += [pip+b.MORE_PINS['nvidia'],pip+b.ALGOS.PINS['nvidia'],pip+b.NEURAL_PINS]
 if b.ALGOS.RAPIDS_EXTRA.get('nvidia'):groups.append(pip+['--extra-index-url','https://pypi.nvidia.com']+b.ALGOS.RAPIDS_EXTRA['nvidia'])
 for i,cmd in enumerate(groups):
  with (T/f'install-{i}.log').open('w') as f:
   try:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1800).returncode
   except subprocess.TimeoutExpired:rc=124
  results.append(dict(command=cmd,returncode=rc));(T/'status.json').write_text(json.dumps(dict(phase='INSTALLING',results=results,time=time.time()),indent=2))
 (R/'opponents-ready.json').write_text(json.dumps(dict(results=results,failed=sum(x['returncode']!=0 for x in results),time=time.time()),indent=2))
except BaseException as e:
 (T/'error.log').write_text(traceback.format_exc());(T/'status.json').write_text(json.dumps(dict(phase='FAILED',error=repr(e),results=results,time=time.time())))
finally:
 log=(O/'repair-controller.log').open('ab');p=subprocess.Popen(['python3',str(R/'repair-worker.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True);(O/'repair-launch.json').write_text(json.dumps(dict(pid=p.pid,time=time.time())));(T/'setup.exit').write_text('0\n' if (R/'opponents-ready.json').exists() else '1\n')
