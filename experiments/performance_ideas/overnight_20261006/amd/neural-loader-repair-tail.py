import pathlib,subprocess,json,sys,time,os
R=pathlib.Path('/root/overnight-ab');H=pathlib.Path('/root/opponent-harness-neural-59fd136091');O=R/'results/opponents';O.mkdir(exist_ok=True)
deadline=time.time()+1800
for marker in ['opponents-data-ready','opponents-env-ready']:
 while not (R/marker).exists():
  if time.time()>deadline:
   (O/'status.json').write_text(json.dumps(dict(status='FAILED',reason='setup marker timeout',marker=marker,updated_at=time.time())))
   raise SystemExit(124)
  (O/'status.json').write_text(json.dumps(dict(status='WAITING_SETUP',marker=marker,updated_at=time.time())))
  time.sleep(30)
wrapper=O/'gpu_only_board.py';wrapper.write_text('import sys,json,pathlib\nsys.path.insert(0,"/root/opponent-harness-neural-59fd136091/tools")\nimport bench_board as b\nimport importlib.util\nmarker=json.loads(pathlib.Path(\'/root/overnight-ab/neural-loader-repair-selection.json\').read_text())\nspec=importlib.util.spec_from_file_location(\'amd_selective_neural_repair\',\'/root/overnight-ab/selective-neural-repair.py\')\nadapter=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(adapter)\nold=b.plan_races\ndef plan(*a,**kw):\n result=[]\n for r in old(*a,**kw):\n  opponents=[x for x in r["opponents"] if not b._is_cpu_arm(x)]\n  if opponents:result.append(dict(r,opponents=opponents,arms=opponents,our_arms={}))\n return result\nb.plan_races=plan\noriginal_run=b.run_race\ndef run_race(*a,**kw):\n root=pathlib.Path("/root/overnight-ab")\n queue=json.loads((root/"repair-queue.json").read_text())\n result=root/"results/repairs/results.json"\n done=json.loads(result.read_text()) if result.exists() else {}\n if any(x["key"] not in done for x in queue):raise SystemExit(75)\n return adapter.repair_neural(a[0],a[1],original_run,marker)\nb.run_race=run_race\nb.main()\n')
cmd=['/root/opponent-venv/bin/python','-u',str(wrapper),'--vendor','amd','--modes','identical','--families','trees,classical,classical2,algos,neural','--rows','full','--neural-shape','full','--rounds','1','--python-env','/root/opponent-venv/bin/python','--skip-install','--no-smoke-gate','--opponents-only','--skip-failed','--cache','/root/opponent-cache','--data-root','/root/datasets/gbm-bench','--opponent-store',str(O/'opponent-store.jsonl'),'--out',str(O/'board')]
selection=json.loads((R/'neural-loader-repair-selection.json').read_text())
cmd+=['--rerun',','.join(selection['races']),'--rerun-before',selection['before']]
(O/'status.json').write_text(json.dumps(dict(status='RUNNING',command=cmd,updated_at=time.time())))
with (O/'board.log').open('a') as f:rc=subprocess.run(cmd,cwd=H,env=os.environ|{'PYTHONDONTWRITEBYTECODE':'1','LD_LIBRARY_PATH':'/opt/rocm/lib'},stdout=f,stderr=subprocess.STDOUT).returncode
(O/'status.json').write_text(json.dumps(dict(status='COMPLETE' if rc==0 else 'YIELDED_TO_CANDIDATE' if rc==75 else 'FAILED',returncode=rc,updated_at=time.time())))
raise SystemExit(rc)
