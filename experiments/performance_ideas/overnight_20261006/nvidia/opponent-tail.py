import json,pathlib,subprocess,os,sys,time
R=pathlib.Path('/root/overnight-nvidia');O=pathlib.Path('/root/campaign-results/gpu-opponents');O.mkdir(exist_ok=True,parents=True)
cfg=json.loads((R/'tail-config.json').read_text());ready=json.loads((R/'opponents-ready.json').read_text())
if ready.get('failed'):raise SystemExit('opponent dependency staging failed')
cmd=['/root/opponent-venv/bin/python','-u',str(R/'gpu-only-board.py'),'--vendor','nvidia','--modes','identical','--families','trees,classical,classical2,algos,neural','--rows','full','--neural-shape','full','--rounds','1','--python-env','/root/opponent-venv/bin/python','--skip-install','--no-smoke-gate','--opponents-only','--skip-failed','--shard',cfg['shard'],'--cache','/root/opponent-cache','--ctd-data','/root/opponent-cache/ctd-data','--more-data','/root/opponent-cache/more-data','--algos-data','/root/opponent-cache/algos-data','--data-root','/root/datasets/gbm-bench','--opponent-store',str(O/'opponent-store.jsonl'),'--out',str(O/'board')]
selection=R/'opponent-repair-selection.json';marker=R/'implicit-env-ready.json'
if selection.exists():
 repair=json.loads(selection.read_text());cmd += ['--rerun',','.join(repair['prefixes']),'--rerun-before',repair['before']]
elif marker.exists():
 cmd += ['--rerun','algos/als/','--rerun-before',json.loads(marker.read_text())['rerun_before']]
env={k:v for k,v in os.environ.items() if not k.startswith('MOJOLEARN_') and k not in ['LD_PRELOAD','LD_LIBRARY_PATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']};env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']='/root/opponent-harness/python'
repair=O/'harness-repair.json'
s=dict(harness_repairs=json.loads(repair.read_text()) if repair.exists() else {},phase='RUNNING',pid=os.getpid(),time=time.time(),command=cmd,harness_source_sha='b0f5242d244ef75cd9ba7d78ca92e10008d75510',policy='GPU opponents only; current hardware receipts, no old RTX4090 times')
(O/'status.json').write_text(json.dumps(s,indent=2))
with (O/'board.log').open('a') as f:p=subprocess.Popen(cmd,cwd='/root/opponent-harness',env=env,stdout=f,stderr=subprocess.STDOUT)
s['board_pid']=p.pid;(O/'status.json').write_text(json.dumps(s,indent=2));rc=p.wait();s.update(phase='YIELD_TO_CANDIDATES' if rc==75 else 'FINISHED' if rc==0 else 'FAILED',returncode=rc,finished=time.time());(O/'status.json').write_text(json.dumps(s,indent=2));sys.exit(rc)
