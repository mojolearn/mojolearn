import argparse, hashlib, json, os, pathlib, signal, subprocess, sys, time

ROOT = pathlib.Path('/root/mojolearn')
OUT = pathlib.Path(os.environ.get('MOJOLEARN_NEURAL_MEASURE_OUT','/root/measure-neural-bc11389'))
PY = '/root/neural-venv/bin/python'
sys.path.insert(0, str(ROOT/'tools'))
import bench_board_neural as neural
import bench_board_evidence as evidence
import cpu_quota

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str)+'\n')
    tmp.replace(path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--lanes', required=True)
    p.add_argument('--budget', type=int, default=1500)
    p.add_argument('--lane-seconds', type=int, default=250)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--algos-driver',default=str(ROOT/'tools/bench_board_algos.py'))
    a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ)
    for key in cpu_quota.INHERITED_CAPS:
        env.pop(key,None)
    env.pop('PYTHONPATH',None)
    env.pop('MOJOLEARN_ALGOS_SMOKE_ROWS',None)
    env.update(MOJOLEARN_BENCH_INSTALLED='1', MOJOLEARN_NUMERIC_MODE='identical',
               MOJOLEARN_REPO_COMMIT='bc11389e6896f918b95a404b8cf2bb97ad2ace09')
    cpu_quota.apply_cpu_quota(env)
    hardware=evidence.hardware_context('nvidia')
    import mojolearn, numpy
    package=pathlib.Path(mojolearn.__file__).parent
    native={str(x.relative_to(package)):evidence.file_hash(x) for x in package.rglob('*.so')}
    write(OUT/'runtime.json', dict(python=sys.version,numpy=numpy.__version__,package=str(package),
        hardware=hardware,native_sha256=native,affinity=sorted(os.sched_getaffinity(0)),
        cpu_max=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text() if pathlib.Path('/sys/fs/cgroup/cpu.max').exists() else None,
        environment=evidence.runtime_environment(env),policy='one excluded warmup, one scored sample; serial GPU lanes; current defaults; no opponents timed'))
    deadline=time.monotonic()+a.budget
    summary_path=OUT/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for lane in a.lanes.split(','):
        if lane in summary: continue
        remaining=deadline-time.monotonic()
        if remaining<30:
            summary[lane]={'status':'pending-budget','executed':False}
            write(summary_path,summary)
            continue
        dest=OUT/lane
        dest.mkdir(exist_ok=True)
        common=['--lane',lane,'--arms','ours','--rounds','1','--out',str(dest/'raw'),
                '--work',str(dest/'work'),'--ours-python',PY,'--theirs-python',PY,
                '--ready-seconds','30' if a.smoke else '120','--warmup-seconds','30' if a.smoke else '180','--round-seconds','30' if a.smoke else '180']
        if lane in ('adagrad','adam'):
            cmd=[PY,a.algos_driver,'race',*common,'--dataset','synthetic','--data',str(OUT/'data')]
            if a.smoke:cmd+=['--smoke-rows','64']
        else:
            cmd=[PY,str(ROOT/'tools/bench_board_neural.py'),'race',*common,'--shape','small' if a.smoke else 'full','--keep-outputs']
        write(dest/'invocation.json',dict(command=cmd,environment=evidence.runtime_environment(env),
            boundary='measurement harness smoke only, reduced recipe, never performance evidence' if a.smoke else 'existing full-workload recipe, no reduced rows/shapes',started=time.time()))
        started=time.monotonic()
        with (dest/'execution.log').open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                rc=child.wait(timeout=min(a.lane_seconds,remaining-15))
                status='completed' if rc==0 else 'failed'
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
                rc=child.returncode;status='timeout-incomplete'
        records=[]
        for f in (dest/'raw').glob('*.json'):
            try:
                r=json.loads(f.read_text())
                records.append(dict(path=str(f),arms=r.get('arms'),quality=r.get('quality'),shape_record=r.get('shape_record'),
                    scorer_preflight=r.get('scorer_preflight'),independent_quality_reference=r.get('independent_quality_reference')))
            except Exception as exc:records.append(dict(path=str(f),parse_error=str(exc)))
        summary[lane]=dict(status=status,exit_code=rc,elapsed_s=time.monotonic()-started,records=records,
            log=str(dest/'execution.log'),executed=True,smoke_only=a.smoke)
        states=[v.get('status') for record in records for v in (record.get('arms') or {}).values()]
        if status=='completed' and (not states or any(s!='ok' for s in states)):
            summary[lane]['status']='failed-driver-record'
        write(summary_path,summary)
        print(json.dumps(dict(lane=lane,status=status,exit_code=rc,elapsed_s=round(time.monotonic()-started,2))),flush=True)

if __name__=='__main__': main()
