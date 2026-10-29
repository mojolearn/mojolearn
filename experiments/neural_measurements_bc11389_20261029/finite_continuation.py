"""Finite, serial smoke-first continuation. No renewal, rental or defaults."""
import datetime, json, os, pathlib, signal, subprocess, sys, time
ROOT=pathlib.Path('/root/measure-final-neural-bc11389')
PY='/root/neural-venv/bin/python'
sys.path.insert(0,'/root/mojolearn/tools')
import bench_board_neural as bench
import bench_board_evidence as evidence
import cpu_quota
STOP_NEW=datetime.datetime(2026,10,29,17,22,tzinfo=datetime.timezone.utc).timestamp()
HARD_WORK_STOP=datetime.datetime(2026,10,29,17,27,tzinfo=datetime.timezone.utc).timestamp()
status={'scope':'measurement smokes first; only sampler full cells admitted by sampler smoke; optimizer reference-only repair',
        'stop_new_utc':'2026-10-29T17:22:00Z','hard_work_stop_utc':'2026-10-29T17:27:00Z',
        'no_renewal':True,'pid':os.getpid(),'phases':[]}
def save():evidence.write(ROOT/'continuation.json',status)
def run(name,cmd,env,budget):
    if time.time()>=STOP_NEW:
        status['phases'].append({'phase':name,'status':'pending-deadline'});save();return False
    budget=max(1,min(budget,int(HARD_WORK_STOP-time.time())))
    row={'phase':name,'command':cmd,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'budget_s':budget}
    status['phases'].append(row);save()
    with (ROOT/(name+'.log')).open('w') as log:
        p=subprocess.Popen(cmd,env=env,cwd='/root/mojolearn',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        row['pid']=p.pid;save()
        try:row['exit_code']=p.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            row.update(status='timeout-incomplete',exit_code=p.returncode)
    row.setdefault('status','completed' if row['exit_code']==0 else 'failed')
    row['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    return row['exit_code']==0

ROOT.mkdir(exist_ok=True);save()
env=dict(os.environ);cpu_quota.apply_cpu_quota(env)
env.update(MOJOLEARN_BOARD_LM_CHECKPOINTS='/root/measure-sampler-bc11389/checkpoints',
           MOJOLEARN_NUMERIC_MODE='identical',MOJOLEARN_BENCH_INSTALLED='1')
env.pop('PYTHONPATH',None)
lanes=['adagrad','adam']+[x for x in bench.LANES if bench.DEVICE_OF[x]!='cpu']
smoke_env=dict(env,MOJOLEARN_NEURAL_MEASURE_OUT=str(ROOT/'family-smoke'))
run('family-smoke',[PY,str(ROOT/'controller.py'),'--smoke','--lanes',','.join(lanes),
    '--budget','800','--lane-seconds','40','--algos-driver','/root/mojolearn/tools/bench_board_algos_quality_repair.py'],smoke_env,830)
sampler_env=dict(env,MOJOLEARN_SAMPLER_OUT=str(ROOT/'sampler-smoke'))
run('sampler-smoke',[PY,str(ROOT/'sampler_ab.py'),'--smoke','--budget','240'],sampler_env,270)
smoke_path=ROOT/'sampler-smoke/summary.json'
smokes=json.loads(smoke_path.read_text()) if smoke_path.exists() else []
sampler_pass=(len(smokes)==8 and all(x.get('status')=='measured' and x.get('smoke_only') is True
    and x.get('quality',{}).get('status')=='scored' and x.get('quality',{}).get('finite_logits') is True for x in smokes))
status['sampler_smoke_pass']=sampler_pass;save()
if sampler_pass:
    run('sampler-full-remaining',[PY,str(ROOT/'sampler_ab.py'),'--budget','360'],
        dict(env,MOJOLEARN_SAMPLER_OUT=str(ROOT/'sampler-full')),390)
else:
    status['phases'].append({'phase':'sampler-full-remaining','status':'blocked-sampler-smoke','new_full_scores_executed':False});save()
run('optimizer-reference-repair',[PY,str(ROOT/'rescore_optimizers.py')],
    dict(env,MOJOLEARN_OPTIMIZER_QUALITY_OUT=str(ROOT/'optimizer-quality-repair')),120)
status['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
status['status']='finished-with-explicit-partial-coverage';save()
print(json.dumps({'status':status['status'],'sampler_smoke_pass':sampler_pass,
                  'phases':[{k:p.get(k) for k in ('phase','status','exit_code')} for p in status['phases']]}),flush=True)
