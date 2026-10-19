import pathlib,subprocess,shutil,json,hashlib
root=pathlib.Path('/Users/andrewhendel/mojolearn-wt/verify-neural-20261012')
p=pathlib.Path(__file__).resolve().parent
out=root/'bench/results/identity_break/2026-10-12-verify-repairs'
python='/tmp/mojolearn-pip-verify-sWrlqV/repair-venv/bin/python'
harness='/tmp/mojolearn-pip-verify-sWrlqV/venv/lib/python3.14/site-packages/mojolearn/_identity_break.py'
reports=[('apple','routine',p.parents[1]/'pip_verify/20261012-205427/full.json'),('apple','neural',p/'apple-neural-baseline.json'),('nvidia','neural',p/'runpod/neural.json'),('amd','neural',p/'do/neural.json')]
manifest=[]
for vendor,scope,source in reports:
 dest=out/vendor;dest.mkdir(parents=True,exist_ok=True)
 target=dest/(scope+'.from-verify.json')
 cmd=[python,str(root/'tools/identity_columns.py'),'from-verify',str(source),'--harness',harness,'--out',str(target)]
 with (p/(vendor+'-'+scope+'-adaptor.log')).open('w') as log:r=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT)
 print(vendor,scope,'adapter_exit',r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
 manifest.append(dict(vendor=vendor,scope=scope,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),record=str(target.relative_to(root))))
for vendor,source in [('apple',p/'apple-repaired-column.json'),('nvidia',p/'runpod/repaired-column.json'),('amd',p/'do/repaired-column.json')]:
 dest=out/vendor/'repaired.identical.json';shutil.copy2(source,dest)
 manifest.append(dict(vendor=vendor,scope='repaired',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),record=str(dest.relative_to(root))))
(out/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('RECORDS_READY',out)
