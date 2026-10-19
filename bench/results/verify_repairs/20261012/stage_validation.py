import pathlib,json,subprocess,shlex
p=pathlib.Path(__file__).resolve().parent;root=pathlib.Path('/Users/andrewhendel/mojolearn-wt/verify-neural-20261012')
for provider in ['do','runpod']:
 s=json.loads((p/provider/'state.json').read_text());ssh=['ssh',*s['ssh']]
 subprocess.run(ssh+['test -f /root/ml-verify/artifacts/repaired-column.exit'],check=True,timeout=30)
 r=subprocess.run(ssh+["/root/ml-verify/venv/bin/python -I -c 'import mojolearn,pathlib;print(pathlib.Path(mojolearn.__file__).parent)'"],capture_output=True,text=True,check=True,timeout=30);pkg=r.stdout.strip()
 for name in ['_verify_reference.py','_verify_all.py','verify_reference/table.json']:
  subprocess.run(ssh+['cat > '+shlex.quote(pkg+'/'+name)],input=(root/'python/mojolearn'/name).read_bytes(),check=True,timeout=60)
 subprocess.run(ssh+['cat > /root/ml-verify/validation.sh'],input=(p/'remote-validation.sh').read_bytes(),check=True)
 r=subprocess.run(ssh+['nohup bash /root/ml-verify/validation.sh > /root/ml-verify/artifacts/validation-job.log 2>&1 < /dev/null & echo $!'],capture_output=True,text=True,check=True)
 print(provider,'VALIDATION_STARTED',r.stdout.strip(),flush=True)
