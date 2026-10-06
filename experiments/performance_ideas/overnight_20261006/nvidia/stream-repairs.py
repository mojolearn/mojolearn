"""Capture ready native repair artifacts and serialize their runs on owned GPUs."""
import hashlib,json,pathlib,subprocess,shlex,time,traceback,os
E=pathlib.Path(__file__).resolve().parent;CPU=['ssh','-i',str(pathlib.Path.home()/'.ssh/id_ed25519'),'-o','BatchMode=yes','-o','ConnectTimeout=20','root@167.99.116.166'];THREAD='01a10f60-3998-7c91-845d-552e1c04d795';LOCAL=E/'repair-artifacts';LOCAL.mkdir(exist_ok=True);sent=set()
def atom(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def notify(key,message):
 if key in sent:return
 p=subprocess.run(['/opt/homebrew/bin/codex','queue','--thread',THREAD,'--message',message],capture_output=True,text=True,timeout=30)
 if p.returncode==0:sent.add(key)
def cmd(argv,**kw):return subprocess.run(argv,capture_output=True,timeout=180,check=True,**kw)
DEFAULT_ENVIRONMENT=json.loads((E/'driver-environment-defaults.json').read_text())
def effective_environment(ident,env):
 defaults=DEFAULT_ENVIRONMENT.get(ident)
 return {k:str(env.get(k,v)) for k,v in defaults.items()} if defaults is not None else {k:str(v) for k,v in env.items()}
def queue_install_command():
 tmp='/root/overnight-nvidia/repair-queue.incoming-'+str(os.getpid())+'-'+str(time.time_ns())
 return 'cat > '+tmp+' && mv '+tmp+' /root/overnight-nvidia/repair-queue.json'
def cases(id,source_sha=""):
 if id=="I06" and source_sha.startswith("e80a1d0a"):
  return [dict(AB_LENGTH=str(l),AB_HEADS="8",AB_KV_HEADS="4") for l in [1024,1536]]
 if id in ['I03','I05','N03','N05']:return [dict(AB_M='257',AB_N='259',AB_K='1025'),dict(AB_M='1023',AB_N='1025',AB_K='2049')]+([dict(AB_M='1025',AB_N='513',AB_K='1025')] if id!='N03' else [])
 if id in ['I02','I04']:return [dict(AB_M='1024',AB_N='1024',AB_K='2048'),dict(AB_M='1023',AB_N='1025',AB_K='2049')]
 if id in ['I06','I07']:return [dict(AB_LENGTH='1024',AB_HEADS='12',AB_KV_HEADS='4'),dict(AB_LENGTH='1536',AB_HEADS='12',AB_KV_HEADS='4')]
 if id=='I08':return [dict(AB_LENGTH='513',AB_CASE='1'),dict(AB_LENGTH='1025',AB_CASE='1')]
 if id=='I09':return [dict(AB_LENGTH='512',AB_FEATURES='64'),dict(AB_LENGTH='1025',AB_FEATURES='64')]
 if id=='I10':return [{}]
 if id=='I11':return [dict(AB_ROWS='65536',AB_VOCAB='4096',AB_FEATURES='64'),dict(AB_ROWS='131071',AB_VOCAB='8192',AB_FEATURES='32')]
 if id=='I13':return [dict(AB_ROWS='100000',AB_DEGREE='32'),dict(AB_ROWS='131071',AB_DEGREE='16')]
 if id=='I14':return [dict(AB_ROWS='100000'),dict(AB_ROWS='131071')]
 if id=='I15':return [dict(AB_ROWS='100000',AB_QUERIES='2000',AB_FEATURES='8',AB_K='8'),dict(AB_ROWS='131071',AB_QUERIES='1024',AB_FEATURES='17',AB_K='16')]
 if id=='I20':return [dict(AB_ROWS='100000',AB_QUERIES='1024',AB_FEATURES='8'),dict(AB_ROWS='131071',AB_QUERIES='513',AB_FEATURES='17'),dict(AB_ROWS='100000',AB_FEATURES='9'),dict(AB_ROWS='100000',AB_FEATURES='17')]
 if id=='I24':return [dict(AB_ROWS='1000000',AB_CLASSES='33'),dict(AB_ROWS='1048573',AB_CLASSES='17'),dict(AB_ROWS='1000001'),dict(AB_ROWS='1048577')]
 if id=='N06':return [{}]
 if id=='N08':return [dict(AB_ROWS='1024',AB_COLUMNS='4096',AB_FEATURES='8'),dict(AB_ROWS='1023',AB_COLUMNS='4097',AB_FEATURES='17')]
 # Exact counterpart fixtures; implicit defaults already measured are deduplicated.
 if id=='A04':return [dict(AB_GROUPS=str(x)) for x in [4096,65537,1048576]]
 if id in ['A07','A08','I18']:return [dict(AB_ROWS=str(n),AB_FEATURES=str(d)) for n,d in [(100000,32),(100001,33),(65537,17)]]
 if id=='I16':return [dict(AB_ROWS='100000',AB_FEATURES=str(d),AB_QUERIES='128') for d in [7,33,65]]
 if id in ['I17','I21']:return [dict(AB_ROWS=str(n),AB_FEATURES=str(d)) for n,d in [(10000,17),(10001,18),(32769,9)]]
 if id=='I19':return [dict(AB_ROWS=str(n)) for n in [65537,98305,131071]]
 if id=='I22':return [dict(AB_ROWS=str(n),AB_FEATURES=str(d)) for n,d in [(65537,33),(65539,34),(131073,17)]]
 if id=='I23':return [dict(AB_OBSERVATIONS=str(n)) for n in [4096,4097,8193]]
 return [dict(AB_ROWS='100000',AB_FEATURES='32'),dict(AB_ROWS='131071',AB_FEATURES='17')]
def normalize(route):
 p=E/(route+'-capture/artifacts/repairs/results.json')
 if not p.exists():return
 raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest();snap=E/(route+'-repair-snapshots')/(digest+'.json');snap.parent.mkdir(exist_ok=True)
 if not snap.exists():snap.write_bytes(raw)
 results=json.loads(raw);rows=[]
 def common(r,case):
  m=r['hardware'];s=r['source_sha'];return dict(effective_environment=effective_environment(r['candidate_id'],r['environment']),id=r['candidate_id'],vendor='nvidia',route=route,case=case,scope=r['scope'],machine=m,baseline_machine=m,candidate_machine=m,source_sha=s,baseline_source_sha=s,candidate_source_sha=s,evidence=str(snap),warmups=1,scored_samples=1,warmup_scope='same_process',status='MEASURED',returncode=0,limitation='Representative production operation; synthetic inputs, not full board dataset qualification.',promotion=False)
 def put(b,c,bns,cns,case):
  if not bns or not cns:return
  row=common(c,case)
  if c['candidate_id']=='I06' and c['source_sha'].startswith('cbcc8dcd3'):row.update(comparison_kind='confounded_schedule_bundle',limitation='Head-reuse flag also changed the requested backward kvgrid schedule in the original timing driver. Bundled measurement only; isolated toggle rerun uses corrected freeze.',promotion=False)
  row.update(artifact_hashes=dict(baseline=b['binary_sha256'],candidate=c['binary_sha256']),baseline_ms=bns/1e6,candidate_ms=cns/1e6);rows.append(row)
 for key,r in results.items():
  if r['candidate_id']=='I23':
   row=common(r,key);row.update(status='NO_DISTINCT_RUNTIME_ARM',returncode=0,limitation='ARIMA_FAST_BATCH_GRAD is guarded by has_apple_gpu_accelerator(); Linux flag arms share the same sequential route. Raw timing retained without ratios.');rows.append(row);continue
  if r['candidate_id']=='I06' and r['source_sha'].startswith('e80a1d0a') and int(effective_environment('I06',r['environment'])['AB_HEADS'])//int(effective_environment('I06',r['environment'])['AB_KV_HEADS'])%2!=0:
   row=common(r,key);row.update(status='NO_DISTINCT_RUNTIME_ARM',returncode=0,limitation='Query/KV head ratio3 misses shared-head reuse guard requiring an even ratio. Existing binary measured again only on new8/4 fixture.');rows.append(row);continue
  if r['candidate_id']=='I07' and r['source_sha'].startswith('cbcc8dcd3'):
   row=common(r,key);row.update(status='NO_DISTINCT_RUNTIME_ARM',returncode=0,limitation='Original timing driver selects no _estash route; both flags ran_arm1030 kept_cells0. Raw timing retained; corrected selector driver pending.');rows.append(row);continue
  if r['candidate_id']=='I15':
   row=common(r,key);row.update(status='NO_DISTINCT_RUNTIME_ARM',returncode=0,limitation='KNN_CERTIFIED_MMA is compile-time Apple-only; NVIDIA macro arms use the same route. Raw timings retained without A/B comparison.');rows.append(row);continue
  if r['candidate_id']=='I19' and r['status']!='MEASURED' and r['arm'] in ['baseline','incumbent'] and int(effective_environment('I19',r['environment']).get('AB_ROWS','0'))>131072:
   row=common(r,key);row.update(status='UNSUPPORTED_BASELINE_SHAPE',returncode=r.get('returncode',1),limitation='Retained rank baseline supports at most4096 rows per segment; original totalrows with32segments exceeded its declared domain. Raw failure preserved; shared admissible shapes appended.');rows.append(row);continue
  if r['status']!='MEASURED':row=common(r,key);row.update(status=r['status'],returncode=r.get('returncode',1));rows.append(row);notify('failure-'+key,'NVIDIA repair measurement failed '+key+'; inspect '+str(p.parent/key/'receipt.json')+' and bounded measurement.log. No identity retests.');continue
  measurements=r['measurements']
  paired=[x for x in measurements if 'baseline_completion_ns' in x and 'candidate_completion_ns' in x]
  for ix,x in enumerate(paired):put(r,r,x['baseline_completion_ns'],x['candidate_completion_ns'],r['key']+'/paired'+str(ix))
  if r.get('paired') and not paired:
   bs=[x for x in measurements if str(x.get('arm'))=='0'];cs=[x for x in measurements if str(x.get('arm'))=='1']
   for ix,(b,c) in enumerate(zip(bs,cs)):put(r,r,b.get('elapsed_ns'),c.get('elapsed_ns'),r['key']+'/runtime'+str(ix))
 for key,c in results.items():
  if c['candidate_id'] in ['I15','I23'] or (c['candidate_id']=='I07' and c['source_sha'].startswith('cbcc8dcd3')):continue
  if c['candidate_id']=='I06' and c['source_sha'].startswith('e80a1d0a') and int(effective_environment('I06',c['environment'])['AB_HEADS'])//int(effective_environment('I06',c['environment'])['AB_KV_HEADS'])%2!=0:continue
  if c['status']!='MEASURED' or c.get('paired') or c['arm'] in ['baseline','incumbent','current','current256','production_default512']:continue
  candidates=[b for b in results.values() if b['status']=='MEASURED' and b['candidate_id']==c['candidate_id'] and b['source_sha']==c['source_sha'] and effective_environment(b['candidate_id'],b['environment'])==effective_environment(c['candidate_id'],c['environment']) and b['arm'] in ['baseline','incumbent','current','current256','production_default512']]
  if not candidates:continue
  b=candidates[0]
  for ix,(bm,cm) in enumerate(zip(b['measurements'],c['measurements'])):
   if c['candidate_id']=='I17' and c['arm'] in ['candidate','lg_exact_off'] and cm.get('policy')=='Depthwise':
    row=common(c,c['key']+'/case'+str(ix));row.update(status='NO_DISTINCT_RUNTIME_ARM',returncode=0,policy='Depthwise',limitation='Resident frontier and exact-batch controls affect Lossguide only; this Depthwise control has the same runtime route. Inherited-partition controls remain distinct. Raw timing preserved.');rows.append(row);continue
   put(b,c,bm.get('elapsed_ns'),cm.get('elapsed_ns'),c['key']+'/case'+str(ix))
 atom(E/(route+'-repair-normalized-measurements.json'),dict(rows=rows,updated=time.time(),snapshot=str(snap)))
while True:
 try:
  discover="""import pathlib,json
root=pathlib.Path('/root/measurement-builder/artifacts');rows=[]
for group in ['nvidia-family-repairs','amd-remaining','nvidia-more','nvidia-priority','a07a08-nvidia','nvidia-corrected','nvidia-corrected-i06']:
 for p in (root/group).glob('*.json'):
  try:d=json.loads(p.read_text())
  except Exception:continue
  if d.get('vendor')=='nvidia' and 'returncode' in d:
   d['_meta_path']=str(p);d['_binary_path']=str(p.with_suffix(''));rows.append(d)
print(json.dumps(rows))
"""
  metas=json.loads(cmd([*CPU,'python3 -c '+shlex.quote(discover)]).stdout);staged=[]
  atom(E/'repair-build-observations.json',dict(time=time.time(),builds=metas))
  for meta in metas:
   if meta['returncode']!=0:notify('build-failed-'+meta['_meta_path'], 'Actual NVIDIA measurement-driver build failed rc='+str(meta['returncode'])+' '+meta['_meta_path']+'; inspect bounded build diagnostics and fix only affected driver. Identity remains accepted.')
  for meta in metas:
   if meta['returncode']!=0:continue
   name=meta.get('name',meta.get('job',pathlib.Path(meta['_binary_path']).name));ident=meta.get('id',name.split('-')[0]);arm=meta.get('arm',name.split('-',1)[1] if '-' in name else 'paired')
   ids=['I03','I05','N03','N05'] if name=='paired' else [ident]
   for ident in ids:
    route='specific' if ident.startswith('N') else 'default';cfg=json.loads((E/(route+'-owner/config.json')).read_text());ssh=['ssh',*cfg['ssh']]
    job=pathlib.Path(meta['_binary_path']).name+'-'+meta['source_sha'][:10];queuefile=E/(route+'-repair-queue.json');queue=json.loads(queuefile.read_text()) if queuefile.exists() else [];new=[]
    for i,env in enumerate(cases(ident,meta['source_sha'])):
     if name=='paired':env=dict(env,AB_KIND=str({'I05':0,'N03':1,'I03':2,'N05':3}[ident]))
     effective=effective_environment(ident,env)
     binary_hash=meta.get('binary_sha256',meta.get('sha256'))
     if any(q['candidate_id']==ident and q['arm']==arm and q['binary_sha256']==binary_hash and effective_environment(ident,q['environment'])==effective for q in queue):continue
     shape=hashlib.sha256(json.dumps(effective,sort_keys=True).encode()).hexdigest()[:10]
     key=ident+'-'+arm+'-'+meta['source_sha'][:10]+'-shape'+shape
     new.append(dict(key=key,candidate_id=ident,arm=arm,job=job,source_sha=meta['source_sha'],source=meta.get('source',''),defines=meta.get('defines',[]),route=route,accelerator='sm_89',environment=env,binary_sha256=meta.get('binary_sha256',meta.get('sha256')),timeout_seconds=1800,paired=name=='paired' or ident in ['I02','N06','N08','A04','I16'],scope='component' if ident in ['I02','I03','I04','I05','N03','N05','N06','N08','A04'] else 'public_caller_component'))
    if not new:continue
    local=LOCAL/job
    if not local.exists():cmd(['rsync','-az','-e',shlex.join(CPU[:-1]),CPU[-1]+':'+meta['_binary_path'],str(local)])
    if hashlib.sha256(local.read_bytes()).hexdigest()!=new[0]['binary_sha256']:raise RuntimeError('repair artifact hash mismatch '+job)
    cmd(['rsync','-az','-e',shlex.join(ssh[:-1]),str(local),ssh[-1]+':/root/overnight-nvidia/bin/'+job]);queue.extend(new);atom(queuefile,queue)
    cmd([*ssh,queue_install_command()],input=queuefile.read_bytes());staged.extend(x['key'] for x in new)
    # Release staging hold only after actual queued work has been delivered.
    subprocess.run(['python3','/Users/andrewhendel/mojolearn-wt/nvidia-overnight-20261006/tools/runpod_usage_lease.py','release-hold',str(E/(route+'-owner/config.json'))],capture_output=True,timeout=90)
  for route in ['specific','default']:
   normalize(route)
   queuefile=E/(route+'-repair-queue.json')
   if queuefile.exists():
    cfg=json.loads((E/(route+'-owner/config.json')).read_text());cmd(['ssh',*cfg['ssh'],queue_install_command()],input=queuefile.read_bytes())
  atom(E/'stream-status.json',dict(status='WATCHING',pid=os.getpid(),time=time.time(),discovered=len(metas),newly_staged=staged))
 except Exception as e:
  (E/'stream-error.log').write_text(traceback.format_exc());atom(E/'stream-status.json',dict(status='ERROR',pid=os.getpid(),time=time.time(),error=repr(e)));notify('stream-error-'+type(e).__name__,'NVIDIA ready-artifact stager failed; inspect '+str(E/'stream-error.log')+' and GPU idle deadlines. Repair actual infrastructure only.')
 time.sleep(30)
