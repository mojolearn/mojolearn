import argparse,concurrent.futures,json,pathlib,subprocess,time,os,shlex
parser=argparse.ArgumentParser(description='Read-only GPU activity and capture watcher; durable alerts through Codex queue.');parser.add_argument('--root',type=pathlib.Path,required=True);parser.add_argument('--thread',required=True);args=parser.parse_args();E=args.root;THREAD=args.thread;sent=set()
def alert(key,message):
 if key in sent:return
 q=subprocess.run(['/opt/homebrew/bin/codex','queue','--thread',THREAD,'--message',message],capture_output=True,text=True,timeout=45)
 with (E/'watch-events.jsonl').open('a') as f:f.write(json.dumps(dict(key=key,message=message,notify_rc=q.returncode,time=time.time()))+'\n')
 if q.returncode==0:sent.add(key)
def probe(route):
 cfg=json.loads((E/(route+'-owner/config.json')).read_text());base=E/(route+'-capture');row=dict(route=route,pod_id=cfg['pod_id'],time=time.time())
 manager=json.loads((base/'manager-status.json').read_text());row['manager']=manager
 if manager.get('status')=='TERMINATED_VERIFIED':return row
 code="""import pathlib,json,subprocess,os
p=pathlib.Path('/root/campaign-results');s=json.loads((p/'status.json').read_text());pid=s.get('worker_pid',s.get('pid'));s['process_exists']=pathlib.Path('/proc/'+str(pid)).exists();s['gpu_activity']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader'],text=True).strip();d=json.loads((p/'results.json').read_text()) if (p/'results.json').exists() else {};s['failures']=[{'key':k,'status':v['status'],'rc':[x['returncode'] for x in v.get('runs',[])]} for k,v in d.items() if v['status']=='MEASUREMENT_FAILED'];s['measured']=sum(v['status']=='MEASURED_COMPONENT' for v in d.values());print(json.dumps(s))
"""
 q=subprocess.run(['ssh',*cfg['ssh'],'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=45)
 row['probe_rc']=q.returncode
 if q.returncode==0:row['remote']=json.loads(q.stdout)
 else:row['probe_error']=q.stderr[-500:]
 return row
while True:
 rows=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures={pool.submit(probe,r):r for r in ['specific','default']}
  for f,route in [(f,r) for f,r in futures.items()]:
   try:row=f.result()
   except Exception as e:row=dict(route=route,error=repr(e))
   rows.append(row)
 for row in rows:
  route=row['route'];remote=row.get('remote',{});manager=row.get('manager',{})
  if manager.get('status')=='ERROR':alert(route+'-manager-error',f'NVIDIA {route} capture/lease error; inspect {E}/{route}-capture/manager-status.json immediately. Retained orphan deadline still applies.')
  for failure in remote.get('failures',[]):alert(route+'-'+failure['key']+'-failed',f'NVIDIA {route} measurement failed {failure}; repair only affected measurement, no repeat identity/compile validation. Evidence {E}/{route}-capture/artifacts/'+failure['key'])
  if remote.get('phase')=='WAITING_FOR_NEXT_MEASUREMENTS':alert(route+'-drained-'+str(remote.get('completed')),f'NVIDIA {route} completed current queue {remote.get("completed")}/{remote.get("queue_total")}, {remote.get("measured")} measured components, {len(remote.get("failures",[]))} failed. Queue ready for repaired full caller drivers and final missing GPU opponents; 30 minute idle deletion begins after capture. Append promptly or hold while actually staging. {E}')
  if remote.get('phase')=='MEASURING' and not remote.get('process_exists'):alert(route+'-dead-'+str(remote.get('current')),f'NVIDIA {route} active worker missing; inspect {E} and restart owned measurement only.')
  if manager.get('status')=='TERMINATED_VERIFIED':alert(route+'-deleted',f'NVIDIA {route} {row["pod_id"]} TERMINATED_VERIFIED after captured idle interval.')
 (E/'watch-status.json').write_text(json.dumps(dict(pid=os.getpid(),time=time.time(),machines=rows),indent=2)+'\n')
 if all(r.get('manager',{}).get('status')=='TERMINATED_VERIFIED' for r in rows):break
 time.sleep(30)
