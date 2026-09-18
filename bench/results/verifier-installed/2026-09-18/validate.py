from pathlib import Path
import os, subprocess

base=Path(__file__).resolve().parent
root=Path('/Users/andrewhendel/mojolearn-wt/verification-coverage-continuation')
host=next((base/'venv/lib').glob('python*/site-packages/mojolearn/host'))
env=dict(os.environ,PYTHONPATH=str(root/'python'),MOJOLEARN_HOST_DIR=str(host),MOJOLEARN_VERIFY_ALL_DRIFT_LANES='ols',MOJOLEARN_NUMERIC_MODE='identical',MOJOLEARN_CPU_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
python='/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python'
tests=['test_verify_pending.py','test_verify_all.py','test_portable_kernel_models.py','test_verify_reference_admit.py','test_host_surface.py','test_verification_coverage.py']
with (base/'focused-tests.log').open('w') as log:
    subprocess.run([python,'-m','pytest','-q',*[str(root/'python/mojolearn/tests'/name) for name in tests]],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('focused tests PASS',flush=True)
subprocess.run([python,str(base/'review_gp.py')],cwd=base,env=env,check=True)
