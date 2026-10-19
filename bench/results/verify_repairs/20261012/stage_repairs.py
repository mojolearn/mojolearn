import pathlib,json,subprocess,time,shlex
p=pathlib.Path(__file__).resolve().parent
root=pathlib.Path('/Users/andrewhendel/mojolearn-wt/verify-neural-20261012')
for provider in ['do','runpod']:
 s=json.loads((p/provider/'state.json').read_text());ssh=['ssh',*s['ssh']]
 deadline=time.time()+1200
 while time.time()<deadline:
  r=subprocess.run(ssh+['test -f /root/ml-verify/artifacts/job.exit'],capture_output=True,timeout=30)
  if r.returncode==0:break
  time.sleep(10)
 else:raise SystemExit(provider+' baseline not finished')
 r=subprocess.run(ssh+["/root/ml-verify/venv/bin/python -I -c 'import mojolearn,pathlib;print(pathlib.Path(mojolearn.__file__).parent)'"],capture_output=True,text=True,check=True,timeout=30)
 pkg=r.stdout.strip()
 subprocess.run(ssh+['cp '+shlex.quote(pkg+'/_identity_break.py')+' /root/ml-verify/artifacts/published-identity-harness.py'],check=True)
 for source,name in [(root/'tools/identity_break.py','_identity_break.py'),(root/'python/mojolearn/_classical_host.py','_classical_host.py'),(root/'python/mojolearn/_expansion_trees.py','_expansion_trees.py')]:
  subprocess.run(ssh+['cat > '+shlex.quote(pkg+'/'+name)],input=source.read_bytes(),check=True,timeout=30)
 subprocess.run(ssh+['cat > /root/ml-verify/repairs.sh'],input=(p/'remote-repairs.sh').read_bytes(),check=True)
 r=subprocess.run(ssh+['nohup bash /root/ml-verify/repairs.sh > /root/ml-verify/artifacts/repairs-job.log 2>&1 < /dev/null & echo $!'],capture_output=True,text=True,check=True)
 print(provider,'REPAIRS_STARTED',r.stdout.strip(),flush=True)
 for name in ['neural.json','neural.log','neural.exit']:
  with (p/provider/name).open('wb') as f:subprocess.run(ssh+['cat /root/ml-verify/artifacts/'+name],stdout=f,check=True,timeout=60)
