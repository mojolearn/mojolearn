from pathlib import Path
import hashlib,io,json,subprocess,sys,tarfile
r=Path(__file__).parent;root=Path('/Users/andrewhendel/mojolearn-wt/host-attribute-error-repair-20261029');cache=Path.home()/'mojolearn-evidence/resident-sampling-runtime-repair-20261029/cache'
sys.path.insert(0,str(root/'tools'));import evidence_r2,bincache
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
selected={p:'evidence/'+p.name for p in r.iterdir() if p.is_file() and p.name!='retention.json'}
for name in ('build','initial-compile','concurrency-refusal','stage','tests','test-tmp'):
 for p in (r/name).rglob('*'):
  if p.is_file():selected[p]='evidence/'+str(p.relative_to(r))
files={}
for result in (r/'initial-compile/build/result.json',r/'build/result.json'):
 if not result.exists():continue
 for item in json.loads(result.read_text()).get('results',[]):
  if item.get('exit_code')!=0 or not item.get('key'):continue
  rp=cache/'artifacts'/item['key']/'receipt.json'
  if not rp.exists():continue
  record=json.loads(rp.read_text())
  for p in rp.parent.rglob('*'):
   if p.is_file():selected[p]='native/'+item['key']+'/'+str(p.relative_to(rp.parent))
  for p in (cache/'work'/item['key']/'roots').rglob('*'):
   if p.is_file():selected[p]='native-roots/'+item['key']+'/'+p.name
  files.update(record.get('fields',{}).get('source',{}).get('files',{}))
archive=r.parent/'host-attribute-repair-543875-apple.tgz'
with tarfile.open(archive,'w:gz') as out:
 for p,name in sorted(selected.items()):out.add(p,arcname=name)
 for rel,h in files.items():
  p=root/rel;assert sha(p)==h,rel;out.add(p,arcname='source/'+rel)
 for revision in ('c9d8c07def89ba898781d6783259a08c30d4d7d0','543875dbb0b4b334153dbe9b61b3d7a60c177b2c'):
  data=subprocess.check_output(['git','show',revision+':bindings/cfb_spectral.mojo'],cwd=root)
  info=tarfile.TarInfo('attempt-source/'+revision+'/cfb_spectral.mojo');info.size=len(data);out.addfile(info,io.BytesIO(data))
 for rel in ('python/mojolearn/_native_public_attributes.py','python/mojolearn/_native_host_archive_predict.py','python/mojolearn/tests/test_host_attribute_lifetime.py'):
  out.add(root/rel,arcname='source/'+rel)
body=archive.read_bytes();digest=hashlib.sha256(body).hexdigest();key='measurements/host-attribute-error-repair-20261029/'+archive.name
record={'archive':str(archive),'bytes':len(body),'sha256':digest,'r2_key':key,'source_files':len(files),'status':'local-retained','scope':'Two repaired native owners; actual focused results only if their independent exit/coverage summaries pass. Original failed whole verifier remains unchanged.'}
(r/'retention.json').write_text(json.dumps(record,indent=2)+'\n')
creds=evidence_r2._creds();put,_=bincache.r2_request(creds,'PUT',key,data=body);get,back=bincache.r2_request(creds,'GET',key);readback=hashlib.sha256(back).hexdigest()
record.update(put_status=put,get_status=get,readback_sha256=readback,status='retained' if put in (200,201) and get==200 and readback==digest else 'failed')
(r/'retention.json').write_text(json.dumps(record,indent=2)+'\n');assert record['status']=='retained';print(json.dumps({k:record[k] for k in ('status','bytes','sha256','source_files')}))
