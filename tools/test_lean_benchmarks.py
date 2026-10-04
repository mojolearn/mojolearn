import pathlib, subprocess, tempfile, json, shutil, importlib.util, datetime
base=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('lean_benchmarks',base/'lean_benchmarks.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
entries=[(f'bench/results/suite/2026-09-{day:02}/result.json',100) for day in range(1,6)]
entries += [('bench/results/other/2026-10-03/result.json',100),('bench/results/undated/reference.json',100)]
selection=module.select(entries,datetime.date(2026,10,4),14,3,['bench/results/suite/2026-09-01'])
assert {r['path'] for r in selection if not r['included']} == {'bench/results/suite/2026-09-02'}
assert len(selection)==6
root=pathlib.Path(tempfile.mkdtemp(prefix='helper-test-',dir=base))
repo=root/'repo'; repo.mkdir(); wt=root/'wt'
def run(*args, ok=True):
    r=subprocess.run(list(map(str,args)),capture_output=True,text=True)
    if ok: assert r.returncode==0,(args,r.stderr)
    return r
def git(p,*args):return run('git','-C',p,*args).stdout
try:
    git(repo,'init','-q');git(repo,'config','user.email','local-test@example.invalid');git(repo,'config','user.name','Local test');git(repo,'config','commit.gpgsign','false')
    for i in range(1,6):
        p=repo/f'bench/results/suite/2020-01-{i:02}/result.json';p.parent.mkdir(parents=True);p.write_text('{}')
    (repo/'.gitignore').write_text('*.log\n')
    git(repo,'add','.');git(repo,'commit','-qm','Test fixtures')
    helper=['python3',base/'lean_benchmarks.py',wt]
    run(*helper,'test-lane','--repo',repo)
    assert not (wt/'bench/results/suite/2020-01-01').exists()
    assert (wt/'bench/results/suite/2020-01-03/result.json').exists()
    run(*helper,'--refresh','--keep','bench/results/suite/2020-01-01')
    assert (wt/'bench/results/suite/2020-01-01/result.json').exists()
    # New run remains visible with the existing exclusion-only patterns.
    p=wt/'bench/results/suite/2020-01-06/result.json';p.parent.mkdir(parents=True);p.write_text('{}')
    git(wt,'add',str(p));git(wt,'commit','-qm','Add a new result')
    assert not git(wt,'status','--porcelain')
    # Refresh refuses to evict ignored evidence in the run aging out.
    evidence=wt/'bench/results/suite/2020-01-03/raw.log';evidence.write_text('unique evidence')
    r=run(*helper,'--refresh',ok=False)
    assert r.returncode!=0 and 'ignored files' in r.stderr
    assert evidence.read_text()=='unique evidence'
    evidence.unlink()
    run(*helper,'--refresh')
    assert not (wt/'bench/results/suite/2020-01-03').exists()
    assert (wt/'bench/results/suite/2020-01-01/result.json').exists()
    assert p.exists()
    p.write_text('changed')
    r=run(*helper,'--refresh',ok=False)
    assert r.returncode!=0 and 'working changes' in r.stderr and p.read_text()=='changed'
    print('PASS: create, newest-three retention, pin/restore, new-run visibility, refresh aging, pin persistence, ignored-evidence protection, dirty-tree protection')
finally:
    shutil.rmtree(root)
