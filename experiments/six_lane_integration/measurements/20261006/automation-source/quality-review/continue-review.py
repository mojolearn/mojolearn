"""Periodically review newly captured evidence and publish through main tools."""
from pathlib import Path
import fcntl,hashlib,json,os,runpy,subprocess,sys,time,traceback
BASE=Path('/Users/andrewhendel/mojolearn-evidence/six-lane-full-ab-20261006')
REPO=Path('/Users/andrewhendel/CascadeProjects/mojolearn')
OUT=BASE/'quality-review';LOGS=OUT/'automation-logs';LOGS.mkdir(exist_ok=True)
STATUS=OUT/'automation-status.json'
quality_alert_update=runpy.run_path(str(OUT/'quality-alerts.py'))['update']
state={'status':'RUNNING','pid':os.getpid(),'interval_seconds':60,'automatic_commits':False,'numerical_work':False,'cycles':0}
def save(**changes):
 state.update(changes,heartbeat_at=time.time())
 t=STATUS.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(STATUS)
def files():
 paths=list((BASE/'apple/captured').rglob('receipt.json'))+list((BASE/'nvidia-native/capture-attempt-02/artifacts').rglob('receipt.json'))
 paths+=[p for p in (BASE/'apple/opponent-quality/captured').rglob('*.json') if p.name=='result.json' or '/board/raw/' in str(p)]
 paths+=[REPO/'tools/six_lane_publish_campaign.py',REPO/'tools/af_quality.py',OUT/'review-next-reg-resample.py',OUT/'review-new-opponents.py']
 return sorted(p for p in paths if p.is_file())
def signature():
 h=hashlib.sha256()
 for p in files():h.update(str(p).encode());h.update(p.read_bytes())
 return h.hexdigest()
def run(command,label):
 path=LOGS/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+label+'.log')
 with path.open('ab') as log:
  r=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 return {'command':command,'returncode':r.returncode,'log':str(path)}
with (OUT/'automation.lock').open('a') as owner:
 fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
 last=None
 while not (OUT/'STOP_AUTOMATION').exists():
  try:
   current=signature()
   if current!=last:
    with (OUT/'publication.lock').open('a') as publock:
     fcntl.flock(publock,fcntl.LOCK_EX)
     results=[]
     for name in ('review-next-reg-resample.py','review-new-opponents.py'):
      results.append(run([sys.executable,str(OUT/name)],name[:-3]))
      if results[-1]['returncode']:break
     if all(r['returncode']==0 for r in results):
      results.append(run([sys.executable,str(REPO/'tools/six_lane_publish_campaign.py'),'--campaign-root',str(BASE),'--out',str(REPO/'experiments/six_lane_integration/measurements/20261006')],'publish'))
     failed=any(r['returncode']!=0 for r in results)
     save(status='ACTION_REQUIRED' if failed else 'RUNNING',phase='Saved quality or publication failed' if failed else 'Saved quality reviewed and published',last_cycle_results=results,cycles=state['cycles']+1,last_input_fingerprint=current)
     with (LOGS/'cycles.jsonl').open('a') as audit:audit.write(json.dumps(dict(at=time.time(),input_fingerprint=current,results=results))+'\n')
     if not failed:last=current
   else:save()
   quality_alert_update()
  except Exception as exc:
   save(status='ACTION_REQUIRED',error=repr(exc),traceback=traceback.format_exc())
   with (LOGS/'exceptions.log').open('a') as log:log.write(time.ctime()+'\n'+traceback.format_exc()+'\n')
  time.sleep(60)
 save(status='STOPPED',phase='External stop marker observed')
