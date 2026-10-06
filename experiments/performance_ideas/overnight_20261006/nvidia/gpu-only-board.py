import sys,json,pathlib
sys.path.insert(0,'/root/opponent-harness/tools');import bench_board as b
original=b.plan_races
run=b.run_race
def plan(*args,**kwargs):
 result=[]
 for race in original(*args,**kwargs):
  opponents=[a for a in race['opponents'] if not b._is_cpu_arm(a)]
  if opponents:result.append(dict(race,opponents=opponents,arms=opponents,our_arms={}))
 return result
def run_race(ctx,race):
 R=pathlib.Path('/root/overnight-nvidia');q=json.loads((R/'repair-queue.json').read_text());p=pathlib.Path('/root/campaign-results/repairs/results.json');done=json.loads(p.read_text()) if p.exists() else {}
 if any(x['key'] not in done for x in q) or not (R/'CANDIDATES_SEALED').exists():raise SystemExit(75)
 return run(ctx,race)
def arm_envs(args,vendor,out,log):
 p=pathlib.Path('/root/overnight-nvidia/implicit-env-ready.json')
 return {'implicit-gpu':json.loads(p.read_text())['python']} if p.exists() else {}
b.setup_arm_venvs=arm_envs
b.plan_races=plan;b.run_race=run_race
sys.exit(b.main())
