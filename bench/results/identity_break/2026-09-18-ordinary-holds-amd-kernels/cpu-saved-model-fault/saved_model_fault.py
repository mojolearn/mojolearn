import os,subprocess,json,hashlib
from pathlib import Path
R=Path('/Users/andrewhendel/mojolearn-wt/qualify-ordinary-23')
O=Path('/Users/andrewhendel/mojolearn-evidence/ordinary-23-2026-09-18')
B=Path('/Users/andrewhendel/mojolearn-evidence/next-wheel-coverage/cpu-kernel-identity/sabotage')
P='/Users/andrewhendel/mojolearn-wt/release-087-final/.pixi/envs/test/bin/python'
models=sorted((R/'bench/results/identity_break/2026-09-18-ordinary-holds-amd-kernels/captures-kernels').glob('attempt-*/saved-*'))
assert len(models)==6
report=O/'amd-models-cpu-native-fault.json'
env=dict(os.environ,MOJOLEARN_HOST_DIR=str(B),MOJOLEARN_HOST_ALLOW_SABOTAGE='1',MOJOLEARN_NUMERIC_MODE='identical',PYTHONPATH=str(R/'python')+os.pathsep+str(R/'tools'))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MOJOLEARN_CPU_THREADS'):env[k]='1'
cmd=[P,str(R/'tools/classical_host_gate.py'),'check',*[str(p) for p in models],'--expect-mismatch','--every-fixture','--report',str(report)]
with (O/'amd-models-cpu-native-fault.log').open('w') as log:r=subprocess.run(cmd,env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT,timeout=120)
(O/'amd-models-cpu-native-fault-provenance.json').write_text(json.dumps(dict(exit_code=r.returncode,native_sha256=hashlib.sha256((B/'_mojolearn_estimators_host.so').read_bytes()).hexdigest(),command=cmd),indent=2)+'\n')
raise SystemExit(r.returncode)
