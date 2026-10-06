import pathlib,json,subprocess,shlex,time,os,traceback
E=pathlib.Path(__file__).resolve().parent;CPU=['ssh','-i',str(pathlib.Path.home()/'.ssh/id_ed25519'),'-o','BatchMode=yes','-o','ConnectTimeout=20','root@167.99.116.166'];sent=set();THREAD='01a10f60-3998-7c91-845d-552e1c04d795'
def alert(k,msg):
 if k in sent:return
 p=subprocess.run(['/opt/homebrew/bin/codex','queue','--thread',THREAD,'--message',msg],capture_output=True,timeout=30)
 if not p.returncode:sent.add(k)
def run(args,**kw):return subprocess.run(args,capture_output=True,timeout=60,check=True,**kw)
def atom(p,data):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
while True:
 try:
  code="""import pathlib,json,time
B=pathlib.Path('/root/measurement-builder');groups={}
for g in ['nvidia-family-repairs','nvidia-priority','nvidia-more','amd-remaining']:
 p=B/'artifacts'/g/'status.json';groups[g]=json.loads(p.read_text()) if p.exists() else {}
claims={p.name:time.time()-p.stat().st_mtime for p in (B/'claims').glob('*') if time.time()-p.stat().st_mtime<1800}
print(json.dumps(dict(groups=groups,claims=claims)))
"""
  state=json.loads(run([*CPU,'python3 -c '+shlex.quote(code)]).stdout);meta=json.loads((E/'repair-build-observations.json').read_text())['builds'];summary={}
  for route in ['specific','default']:
   cfg=json.loads((E/(route+'-owner/config.json')).read_text());ssh=['ssh',*cfg['ssh']];q=json.loads((E/(route+'-repair-queue.json')).read_text());p=E/(route+'-capture/artifacts/repairs/results.json');results=json.loads(p.read_text()) if p.exists() else {};missing=[x['key'] for x in q if x['key'] not in results];failed=[k for k,v in results.items() if v['status']!='MEASURED']
   groups=state['groups'];claims=state['claims']
   if route=='specific':
    build_ready=groups['nvidia-priority'].get('phase')=='BUILDS_FINISHED' and groups['nvidia-priority'].get('failed')==0 and any(x.get('name')=='paired' and x['returncode']==0 for x in meta)
    blockers=[k for k in claims if k=='nvidia-priority' or k.startswith(('repair-specific','nvidia-specific'))]
    # Original batch still building AMD equivalents cannot add NVIDIA N jobs.
    expected={'N03','N05','N06','N07','N08'}
   else:
    build_ready=all((groups[g].get('phase')=='BUILDS_FINISHED' or groups[g].get('status') in ['FINISHED','BUILDS_FINISHED','COMPLETE']) for g in ['nvidia-family-repairs','nvidia-more','amd-remaining'])
    blockers=list(claims);expected={'I02','I03','I04','I05','I06','I07','I08','I09','I10','I11','I12','I13','I14','I15','I16','I17','I18','I19','I20','I21','I22','I23','I24','A04','A07','A08'}
   seen={x['candidate_id'] for x in q};eligible=build_ready and not blockers and not missing and not failed and expected<=seen
   remote="python3 - <<'X'\nimport pathlib,json\nr=pathlib.Path('/root/overnight-nvidia');o=pathlib.Path('/root/campaign-results');ready=r/'opponents-ready.json';tail=o/'gpu-opponents/status.json';print(json.dumps(dict(ready=json.loads(ready.read_text()) if ready.exists() else {},tail=json.loads(tail.read_text()) if tail.exists() else {},worker=json.loads((o/'status.json').read_text()))))\nX"
   actual=json.loads(run([*ssh,remote]).stdout);eligible=eligible and actual['ready'].get('failed')==0
   if eligible:
    receipt=dict(time=time.time(),route=route,groups=groups,claims=claims,completed=len(results),queue_total=len(q),policy='All applicable candidate jobs captured; candidate arrivals preempt opponent tail between races')
    run([*ssh,'cat > /root/overnight-nvidia/CANDIDATES_SEALED'],input=json.dumps(receipt).encode())
   else:run([*ssh,'rm -f /root/overnight-nvidia/CANDIDATES_SEALED'])
   tail=actual['tail']
   if tail.get('phase')=='FAILED':alert('tail-'+route,'NVIDIA '+route+' GPU-opponent tail failed; inspect captured gpu-opponents/board.log and status.json under '+str(E/(route+'-capture/artifacts')))
   if actual['ready'].get('failed',0):alert('deps-'+route,'NVIDIA '+route+' opponent dependency setup failed; inspect captured opponent-setup logs. Tail is blocked, candidate measurements remain authorized.')
   summary[route]=dict(sealed=eligible,build_ready=build_ready,pending_claims=blockers,missing_measurements=missing,failed_measurements=failed,missing_ids=sorted(expected-seen),remote=actual)
  atom(E/'tail-monitor-status.json',dict(pid=os.getpid(),time=time.time(),routes=summary))
 except Exception as e:
  (E/'tail-monitor-error.log').write_text(traceback.format_exc());alert('monitor-'+type(e).__name__,'NVIDIA tail monitor infrastructure problem: '+str(E/'tail-monitor-error.log')+'; check cloud idle deadlines.')
 time.sleep(30)
