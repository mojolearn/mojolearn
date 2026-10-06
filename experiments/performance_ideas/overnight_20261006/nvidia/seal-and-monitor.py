import pathlib,json,subprocess,shlex,time,os,traceback
E=pathlib.Path(__file__).resolve().parent;CPU=['ssh','-i',str(pathlib.Path.home()/'.ssh/id_ed25519'),'-o','BatchMode=yes','-o','ConnectTimeout=20','root@167.99.116.166'];sent=set();THREAD='01a10f60-3998-7c91-845d-552e1c04d795'
def alert(k,msg):
 if k in sent:return
 p=subprocess.run(['/opt/homebrew/bin/codex','queue','--thread',THREAD,'--message',msg],capture_output=True,timeout=30)
 if not p.returncode:sent.add(k)
def run(args,**kw):return subprocess.run(args,capture_output=True,timeout=60,check=True,**kw)
def atom(p,data):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)

TERMINAL_OWNER_STATUSES={'TERMINATED','TERMINATED_VERIFIED'}

def safe_alert(key, message):
 try:
  alert(key, message)
 except Exception:
  try:
   with (E/'tail-monitor-error.log').open('a') as fh:
    fh.write(str(time.time())+' alert delivery failed\n'+traceback.format_exc())
  except OSError:pass

def record_error(scope):
 try:
  with (E/'tail-monitor-error.log').open('a') as fh:
   fh.write(str(time.time())+' '+scope+'\n'+traceback.format_exc())
 except OSError:pass
 safe_alert('monitor-'+scope,'NVIDIA tail monitor '+scope+' problem: '+str(E/'tail-monitor-error.log')+'; other routes continue, no replacement pods are provisioned.')

def terminal_owner(route):
 p=E/(route+'-capture/manager-status.json')
 if not p.exists():return None
 owner=json.loads(p.read_text())
 if owner.get('status') in TERMINAL_OWNER_STATUSES:return owner
 return None

def monitor_route(route,state,meta):
 # This guard precedes config loading, SSH and worker recovery. The manager's
 # verified retirement is authoritative; this watcher never replaces a pod.
 owner=terminal_owner(route)
 if owner is not None:
  return dict(status='RETIRED',owner_status=owner['status'],owner=owner,
              ssh_skipped=True,seal_action='UNCHANGED_RETIRED')
 cfg=json.loads((E/(route+'-owner/config.json')).read_text());ssh=['ssh',*cfg['ssh']];q=json.loads((E/(route+'-repair-queue.json')).read_text());p=E/(route+'-capture/artifacts/repairs/results.json');results=json.loads(p.read_text()) if p.exists() else {};missing=[x['key'] for x in q if x['key'] not in results];unsupported_shapes=[k for k,v in results.items() if v['candidate_id']=='I19' and v['arm'] in ['baseline','incumbent'] and v['source_sha'].startswith('5b467815') and int(v['environment'].get('AB_ROWS','1000000'))>131072];failed=[k for k,v in results.items() if v['status']!='MEASURED' and k not in unsupported_shapes]
 recovery=json.loads(run([*ssh,'python3 /root/overnight-nvidia/recover-repair-worker.py']).stdout)
 if recovery['status'] in ['RESTARTED','ORPHAN_CHILD_ACTIVE']:safe_alert('worker-'+route+'-'+str(recovery),'NVIDIA '+route+' owned worker liveness: '+json.dumps(recovery)+'. Inspect captured repair-controller.log; completed cells and pending queue preserved.')
 local_claims=E/'repair-claims';local_claims.mkdir(exist_ok=True)
 if state is not None:
  groups=state['groups'];claims=state['claims']
  if route=='specific':
   build_ready=groups['nvidia-priority'].get('phase')=='BUILDS_FINISHED' and groups['nvidia-priority'].get('failed')==0 and any(x.get('name')=='paired' and x['returncode']==0 for x in meta)
   blockers=[k for k in claims if k=='nvidia-priority' or k.startswith(('repair-specific','nvidia-specific'))]
   # Original batch still building AMD equivalents cannot add NVIDIA N jobs.
   expected={'N03','N05','N06','N07','N08'}
  else:
   build_ready=all((groups[g].get('phase')=='BUILDS_FINISHED' or groups[g].get('status') in ['FINISHED','BUILDS_FINISHED','COMPLETE']) for g in ['nvidia-family-repairs','nvidia-more','amd-remaining'])
   blockers=list(claims);expected={'I02','I03','I04','I05','I06','I07','I08','I09','I10','I11','I12','I13','I14','I15','I16','I17','I18','I19','I20','I21','I22','I23','I24','A04','A07','A08'}
  bad_builds=[x['_meta_path'] for x in meta if x['returncode']!=0 and (route=='default' or x.get('id','').startswith('N') or x.get('name')=='paired')]
  build_ready=build_ready and not bad_builds
  blockers += [p.name for p in local_claims.glob(route+'-*') if time.time()-p.stat().st_mtime<1800]
  seen={x['candidate_id'] for x in q};eligible=build_ready and not blockers and not missing and not failed and expected<=seen
 else:
  groups={};claims={};build_ready=None;bad_builds=[];blockers=[];expected=set();seen={x['candidate_id'] for x in q};eligible=None
 remote="python3 - <<'X'\nimport pathlib,json\nr=pathlib.Path('/root/overnight-nvidia');o=pathlib.Path('/root/campaign-results');ready=r/'opponents-ready.json';tail=o/'gpu-opponents/status.json';print(json.dumps(dict(implicit=json.loads((r/'implicit-env-ready.json').read_text()) if (r/'implicit-env-ready.json').exists() else {},ready=json.loads(ready.read_text()) if ready.exists() else {},tail=json.loads(tail.read_text()) if tail.exists() else {},worker=json.loads((o/'status.json').read_text()))))\nX"
 actual=json.loads(run([*ssh,remote]).stdout)
 if actual.get('implicit'):(local_claims/(route+'-implicit-env')).unlink(missing_ok=True)
 if state is not None:eligible=eligible and actual['ready'].get('failed')==0
 if state is None:
  seal_action='UNCHANGED_BUILDER_UNAVAILABLE'
 elif eligible:
  seal_action='SEALED'
  receipt=dict(time=time.time(),route=route,groups=groups,claims=claims,completed=len(results),queue_total=len(q),policy='All applicable candidate jobs captured; candidate arrivals preempt opponent tail between races')
  run([*ssh,'cat > /root/overnight-nvidia/CANDIDATES_SEALED'],input=json.dumps(receipt).encode())
 else:
  seal_action='UNSEALED'
  run([*ssh,'rm -f /root/overnight-nvidia/CANDIDATES_SEALED'])
 tail=actual['tail']
 if tail.get('phase')=='FAILED':safe_alert('tail-'+route,'NVIDIA '+route+' GPU-opponent tail failed; inspect captured gpu-opponents/board.log and status.json under '+str(E/(route+'-capture/artifacts')))
 if actual['ready'].get('failed',0):safe_alert('deps-'+route,'NVIDIA '+route+' opponent dependency setup failed; inspect captured opponent-setup logs. Tail is blocked, candidate measurements remain authorized.')
 return dict(status='MONITORED',builder_available=state is not None,seal_action=seal_action,worker_liveness=recovery,sealed=eligible,build_ready=build_ready,failed_builds=bad_builds,pending_claims=blockers,missing_measurements=missing,failed_measurements=failed,unsupported_baseline_shapes=unsupported_shapes,missing_ids=sorted(expected-seen) if state is not None else None,remote=actual)

def load_builder_state():
 code="import pathlib,json,time\nB=pathlib.Path('/root/measurement-builder');groups={}\nfor g in ['nvidia-family-repairs','nvidia-priority','nvidia-more','amd-remaining']:\n p=B/'artifacts'/g/'status.json';groups[g]=json.loads(p.read_text()) if p.exists() else {}\nclaims={p.name:time.time()-p.stat().st_mtime for p in (B/'claims').glob('*') if time.time()-p.stat().st_mtime<1800}\nprint(json.dumps(dict(groups=groups,claims=claims)))\n"
 state=json.loads(run([*CPU,'python3 -c '+shlex.quote(code)]).stdout)
 meta=json.loads((E/'repair-build-observations.json').read_text())['builds']
 if not isinstance(state.get('groups'),dict) or not isinstance(state.get('claims'),dict):raise ValueError('Builder state missing groups/claims')
 return state,meta

def monitor_once():
 summary={};state=None;meta=[];builder_error=None
 try:
  state,meta=load_builder_state()
 except Exception as exc:
  builder_error=type(exc).__name__+': '+str(exc)
  record_error('builder-unavailable; existing candidate seals unchanged')
 for route in ['specific','default']:
  try:
   summary[route]=monitor_route(route,state,meta)
  except Exception as exc:
   summary[route]=dict(status='MONITOR_ERROR',error=type(exc).__name__+': '+str(exc),seal_action='UNKNOWN_ROUTE_ERROR')
   record_error('route-'+route)
 atom(E/'tail-monitor-status.json',dict(pid=os.getpid(),time=time.time(),builder_available=state is not None,builder_error=builder_error,routes=summary))
 return summary

def main():
 while True:
  try:monitor_once()
  except Exception:record_error('status-persistence')
  time.sleep(30)

if __name__=='__main__':main()
