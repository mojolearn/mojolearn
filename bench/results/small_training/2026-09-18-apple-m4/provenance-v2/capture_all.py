import json, os, pathlib, subprocess, time
root=pathlib.Path('/Users/andrewhendel/mojolearn-wt/next-wheel-coverage')
base=pathlib.Path(__file__).parent
python='/Users/andrewhendel/mojolearn-wt/release-087-final/.pixi/envs/test/bin/python'
clean='/Users/andrewhendel/mojolearn-wt/release-087-final/python/mojolearn/host'
link=root/'python/mojolearn/identical'
assert not link.exists() and not link.is_symlink()
results=[]
for name,backend in [('cpu','cpu'),('cpu-replay','cpu'),('metal','metal'),('cpu-ridge-sabotage','cpu')]:
 env=dict(os.environ,MOJOLEARN_HOST_DIR=clean)
 if name=='cpu-ridge-sabotage':
  env['MOJOLEARN_HOST_DIR']='/Users/andrewhendel/mojolearn-evidence/next-wheel-coverage/small-training-ridge-fault-host'
  env['MOJOLEARN_HOST_ALLOW_SABOTAGE']='1'
 cmd=[python,str(root/'tools/capture_small_training.py'),'--require-backend',backend,'--out',str(base/name)]
 if backend=='metal':
  link.symlink_to('/Users/andrewhendel/mojolearn-wt/release-087-final/python/mojolearn/identical',target_is_directory=True)
 started=time.monotonic()
 try:
  with (base/(name+'.log')).open('w') as log:
   code=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 finally:
  if backend=='metal':link.unlink()
 result=dict(name=name,command=cmd,seconds=round(time.monotonic()-started,6),exit_code=code)
 results.append(result)
 (base/'capture_times.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps(result),flush=True)
 if code:raise SystemExit(code)
