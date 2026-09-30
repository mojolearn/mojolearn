#!/usr/bin/env python3
"""Resume a frozen full-data board plan; queue completion is never coverage completion."""
import argparse,json,os,subprocess,sys,time,signal
from pathlib import Path
import bench_board_watchdog as watch

def write(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def coverage(plan,records,exceptions):
 missing=[rid for rid in plan if rid not in records]
 pending=[rid for rid in missing if rid not in exceptions]
 bad=[]
 for rid,r in records.items():
  if rid not in plan:continue
  if not r.get('cells'):bad.append({'race':rid,'reason':'no measurement cells'})
  if r.get('status')!='done':bad.append({'race':rid,'reason':'race failed'})
  for c in r.get('cells',[])+r.get('infer_cells',[]):
   if c.get('status')!='ok' or c.get('median_ms') is None:bad.append({'race':rid,'arm':c.get('arm'),'reason':c.get('status')})
 return {'expected':len(plan),'recorded':len(set(plan)&records.keys()),'missing':missing,'pending':pending,'exceptions':exceptions,'unusable_cells':bad,'complete':not missing and not bad}

def kill_owned(proc):
 ps=watch.processes();owned={proc.pid}
 while True:
  new={pid for pid,(parent,*_) in ps.items() if parent in owned}-owned
  if not new:break
  owned|=new
 for pid in sorted(owned,reverse=True):
  try:os.kill(pid,signal.SIGKILL)
  except ProcessLookupError:pass
 proc.wait()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--chunk-seconds',type=int,default=2700);a=ap.parse_args()
 root=Path(a.root).expanduser()
 if (root/'blocked.json').exists():raise SystemExit('Resume blocked: '+(root/'blocked.json').read_text())
 cfg=json.loads((root/'resume-config.json').read_text());out=root/'board';out.mkdir(exist_ok=True)
 failures=root/'exceptions.json';exc=json.loads(failures.read_text()) if failures.exists() else {}
 stop=time.monotonic()+a.chunk_seconds
 while True:
  records=json.loads((out/'board.json').read_text()).get('races',{}) if (out/'board.json').exists() else {}
  status=coverage(cfg['plan'],records,exc);write(root/'coverage.json',status)
  if (root/'previous-board.json').exists():
   prior=json.loads((root/'previous-board.json').read_text()).get('races',{})
   combined=coverage(cfg.get('original_plan',cfg['plan']),{**prior,**records},exc)
   combined['provenance']={'prior_board':str(root/'previous-board.json'),'current_board':str(out/'board.json'),'current_source':cfg.get('source_commit'),'note':'Coverage summary across separate wheel records; previous measurements are not relabeled as current.'}
   write(root/'combined-coverage.json',combined)
  if not status['pending']:return 0 if status['complete'] else 2
  if time.monotonic()>=stop:return 75
  rid=status['pending'][0]
  cmd=[sys.executable,str(root/'run-race.py'),'--vendor',cfg['vendor'],'--mojolearn-version',cfg['version'],'--base-python',sys.executable,'--out',str(out),'--cache',str(root/'cache'),'--data-root',str(Path.home()/'datasets/gbm-bench'),'--no-cpu-arm','--no-smoke-gate','--rounds','1','--race-id',rid]
  cmd+=cfg.get('extra_args',[])
  print('RESUME',rid,flush=True)
  with (root/'measurements.log').open('a') as log:
   proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:rc=proc.wait(timeout=int(cfg.get('race_timeout_s',900)))
   except subprocess.TimeoutExpired:kill_owned(proc);rc=124
  records=json.loads((out/'board.json').read_text()).get('races',{}) if (out/'board.json').exists() else {}
  if rid not in records and rc!=124:
   write(root/'blocked.json',{'race':rid,'exit':rc,'reason':'No race result; setup or driver error must be investigated'})
   return 1
  if rid not in records:
   exc[rid]={'exit':rc,'reason':'TIMEOUT: race exceeded wall-clock budget; raw logs preserved; never counted as complete','timeout_s':int(cfg.get('race_timeout_s',900))};write(failures,exc)
  print('RESULT',rid,'exit',rc,'recorded',rid in records,flush=True)
if __name__=='__main__':sys.exit(main())
