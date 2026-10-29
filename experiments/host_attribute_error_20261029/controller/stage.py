from pathlib import Path
import hashlib,json,shutil,subprocess,sys
r=Path(__file__).parent
root=Path('/Users/andrewhendel/mojolearn-wt/host-attribute-error-repair-20261029')
cache=Path.home()/'mojolearn-evidence/resident-sampling-runtime-repair-20261029/cache'
base=Path.home()/'mojolearn-evidence/resident-sampling-runtime-repair-20261029/final-bc11389-venv'
package=base/'lib/python3.13/site-packages/mojolearn'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result=json.loads((r/'build/result.json').read_text())
names={'_mojolearn_dev_tokenizer_host':'_mojolearn_tokenizer_host','_mojolearn_dev_runtime_classical_host':'_mojolearn_runtime_classical_host'}
original=json.loads((r/'initial-compile/build/result.json').read_text())
selected=[x for x in original['results'] if x.get('stem')=='_mojolearn_dev_tokenizer_host'] + [x for x in result['results'] if x.get('stem')=='_mojolearn_dev_runtime_classical_host']
assert len(selected)==2 and all(x['exit_code']==0 for x in selected)
stage=r/'stage/mojolearn';(stage/'host').mkdir(parents=True,exist_ok=True);(stage/'__init__.py').touch()
records=[]
with (r/'stage.log').open('w') as log:
 for item in selected:
  rp=cache/'artifacts'/item['key']/'receipt.json';d=json.loads(rp.read_text());raw=rp.parent/(d['stem']+'.so')
  assert sha(raw)==d['sha256']
  mismatches=[rel for rel,h in d['fields']['source']['files'].items() if sha(root/rel)!=h];assert not mismatches,mismatches
  target=stage/'host'/(names[item['stem']]+'.so');shutil.copy2(raw,target)
  subprocess.run([sys.executable,str(root/'packaging/macos/stage_dylibs.py'),str(target),str(root/'.pixi/envs/default/lib')],stdout=log,stderr=subprocess.STDOUT,check=True)
  records.append({'stem':names[item['stem']],'compile_receipt':str(rp),'compile_receipt_sha256':sha(rp),'raw':str(raw),'raw_sha256':sha(raw),'staged':str(target),'staged_sha256':sha(target),'source_count':len(d['fields']['source']['files']),'source_mismatches':mismatches})
 subprocess.run(['/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/pkg/bin/python',str(root/'packaging/portable_math/stage.py'),str(stage/'.dylibs'),'--helper',str(package/'.dylibs/libMojolearnMath.dylib'),'--receipt',str(r/'portable-stage.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
runtime={p.name:sha(p) for p in (stage/'.dylibs').glob('*.dylib')};assert runtime=={p.name:sha(p) for p in (package/'.dylibs').glob('*.dylib')}
clone=r/'venv-repaired';assert not clone.exists();subprocess.run(['cp','-cR',str(base),str(clone)],check=True)
newpkg=clone/'lib/python3.13/site-packages/mojolearn'
for record in records:
 target=newpkg/'host'/(record['stem']+'.so');record['previous_sha256']=sha(target);shutil.copy2(record['staged'],target);record['installed']=str(target)
python_changes=[]
for name in ('_native_public_attributes.py','_native_host_archive_predict.py'):
 target=newpkg/name;old=sha(target);source=root/'python/mojolearn'/name;shutil.copy2(source,target);python_changes.append({'relative':name,'source':str(source),'previous_sha256':old,'sha256':sha(target)})
changed=[str(p.relative_to(newpkg)) for p in newpkg.rglob('*.so') if sha(p)!=sha(package/p.relative_to(newpkg))]
assert sorted(changed)==sorted('host/'+record['stem']+'.so' for record in records)
(r/'tests').mkdir(exist_ok=True);shutil.copy2(root/'python/mojolearn/tests/test_host_attribute_lifetime.py',r/'tests')
receipt={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'base_install':str(package),'clone':str(clone),'owners':records,'changed_extensions':changed,'runtime_identical':runtime,'python_changes':python_changes,'scope':'Private focused regression overlay; original bc install/payload immutable; no full-release admission'}
(r/'stage-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('staged two owners and two Python setup files; other77 extensions and five runtimes unchanged')
