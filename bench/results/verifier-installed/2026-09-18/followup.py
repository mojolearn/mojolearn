from pathlib import Path
import json, os, subprocess

base=Path(__file__).resolve().parent
root=Path('/Users/andrewhendel/mojolearn-wt/verification-coverage-continuation')
python=str(base/'venv/bin/python')
env={k:v for k,v in os.environ.items() if not k.startswith(('MOJOLEARN_','DYLD_')) and k not in ('PYTHONPATH','PYTHONHOME','PYTHONOPTIMIZE')}
env.update(MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_CPU_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
def run(name,cmd,environment=env):
    with (base/(name+'.json')).open('w') as out, (base/(name+'.log')).open('w') as err:
        p=subprocess.run(cmd,cwd=base,env=environment,stdout=out,stderr=err,check=True)
    print(name,'PASS',flush=True)
run('loaded-lm-installed',[python,'-m','mojolearn','verify-causal-lm','--device','cpu','--formats','float32','bfloat16','int8','--output',str(base/'loaded-lm-capture.json')])
for vendor,path in [('cpu','cpu.json'),('apple','metal.json'),('amd','amd/capture.json')]:
    run('loaded-lm-compare-'+vendor,[python,'-m','mojolearn','verify-causal-lm','--compare',str(base/'loaded-lm-capture.json'),str(root/'bench/results/loaded_causal_lm/2026-09-18/v2'/path)])
host=next((base/'venv/lib').glob('python*/site-packages/mojolearn/host'))
sourceenv=dict(env,PYTHONPATH=str(root/'python'),MOJOLEARN_HOST_DIR=str(host))
run('models-count-fixed',[python,'-m','mojolearn','verify','--models-only','--repeats','2','--json'],sourceenv)
r=json.loads((base/'models-count-fixed.json').read_text())
assert r['models_checked']==58 and r['model_lanes_checked']==10
assert r['counts']['IDENTICAL']==116 and not any(r['counts'][k] for k in ('REFUSED','DIVERGENT','OWED'))
testpython='/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python'
tests=['test_verify_pending.py','test_verify_all.py','test_portable_kernel_models.py','test_verify_reference_admit.py']
with (base/'reporting-tests.log').open('w') as log:
    subprocess.run([testpython,'-m','pytest','-q',*[str(root/'python/mojolearn/tests'/name) for name in tests]],cwd=root,env=sourceenv,stdout=log,stderr=subprocess.STDOUT,check=True)
print('reporting tests PASS',flush=True)
subprocess.run([python,str(base/'review_gp.py')],cwd=base,env=sourceenv,check=True)
