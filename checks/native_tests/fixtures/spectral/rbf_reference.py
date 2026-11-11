"""Supplementary RBF input oracle; Float64 NumPy formula, no product import.

Twenty independent row chunks run across os.cpu_count() processes. Matrix
assembly preserves row order and writes only test-reference artifacts.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def chunk(job):
    gamma, first = job
    x = np.frombuffer((HERE/'embedding150-input.f32').read_bytes(),dtype='<f4').reshape(150,2).astype(np.float64)
    distance = np.sum((x[first:first+15,None,:]-x[None,:,:])**2,axis=2)
    expected = np.exp(-gamma*distance).astype('<f4')
    return gamma,first,expected.tobytes()


if __name__ == '__main__':
    jobs = [(gamma,first) for gamma in (0.5,1.0) for first in range(0,150,15)]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        parts = list(pool.map(chunk,jobs))
    artifacts = []
    for gamma in (0.5,1.0):
        data = b''.join(p[2] for p in sorted(parts,key=lambda v:v[1]) if p[0]==gamma)
        assert len(data)==150*150*4
        name = 'rbf150-half-reference.f32' if gamma==0.5 else 'rbf150-one-reference.f32'
        (HERE/name).write_bytes(data)
        artifacts.append({'file':name,'gamma':gamma,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    report = {'status':'supplementary independent reference preparation only; no native execution',
              'input':'embedding150-input.f32','input_sha256':hashlib.sha256((HERE/'embedding150-input.f32').read_bytes()).hexdigest(),
              'shape':[150,2],'formula':'exp(-gamma * sum((Float64(X_i)-Float64(X_j))**2)), rounded once to Float32',
              'gammas':'Embedding default1/features=0.5 and Clustering default1.0 for the original150x2 input.',
              'artifacts':artifacts,'workers':os.cpu_count(),'independent_jobs':len(jobs),'numpy':np.__version__,
              'native_tolerance':{'atol':0.000002,'rtol':0.00002},
              'tolerance_note':'Supplementary bound allows binary32 squared-distance/exp rounding relative to the independent binary64 formula. It changes no original test tolerance.',
              'limits':['No existing Python numerical RBF assertion was identified in the audited Spectral files; no deletion is claimed.','No fitted labels/eigenbasis or release hash is generated.']}
    (HERE/'rbf-reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'artifacts':len(artifacts),'bytes':sum(a['bytes'] for a in artifacts),'jobs':len(jobs),'workers':os.cpu_count()}))
