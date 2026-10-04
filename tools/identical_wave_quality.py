#!/usr/bin/env python3
"""Analytic/spectral quality gates; NumPy is an untimed reference only."""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def eigh():
    from mojolearn import linalg
    rows = []
    for n in (4096, 513):
        # Dense rank-one update of a diagonal matrix: nontrivial rotation,
        # analytically known spectrum, no cubic reference eigensolve.
        u = (((np.arange(n, dtype=np.int64) * 17) % 31) - 15).astype(np.float32)
        u /= np.float32(np.linalg.norm(u.astype(np.float64)))
        matrix = np.eye(n, dtype=np.float32) * np.float32(2) + np.outer(u, u)
        values, vectors = linalg.eigh(matrix)
        w, v = np.asarray(values, dtype=np.float64), np.asarray(vectors, dtype=np.float64)
        a = matrix.astype(np.float64)
        residual = float(np.linalg.norm(a @ v - v * w) / np.linalg.norm(a))
        orthogonality = float(np.linalg.norm(v.T @ v - np.eye(n)) / np.sqrt(n))
        expected = np.full(n, 2.0); expected[-1] = 2.0 + np.dot(u.astype(np.float64), u.astype(np.float64))
        eigenvalue_error = float(np.max(np.abs(np.sort(w) - expected)))
        assert np.isfinite(w).all() and np.isfinite(v).all()
        assert residual < 2e-4 and orthogonality < 2e-4 and eigenvalue_error < 2e-4, (n, residual, orthogonality, eigenvalue_error)
        rows.append(dict(n=n, residual=residual, orthogonality=orthogonality, eigenvalue_error=eigenvalue_error))
    return rows


def pca():
    from mojolearn import PCA
    rows=[]
    for n,d in ((4096,220),(1500,33),(900,7)):
        i=np.arange(n*d,dtype=np.int64).reshape(n,d)
        x=((((i*2654435761)%1000003)-500001).astype(np.float32)/np.float32(977))
        x=np.ascontiguousarray(x*(1+(np.arange(d)%11)).astype(np.float32))
        model=PCA(n_components=d, svd_solver='covariance_eigh', numeric_mode='identical').fit(x)
        centered=x.astype(np.float64)-x.mean(axis=0,dtype=np.float64)
        covariance=centered.T@centered/(n-1)
        w=np.asarray(model.explained_variance_,dtype=np.float64)
        v=np.asarray(model.components_,dtype=np.float64).T
        reference=np.linalg.eigvalsh(covariance)[::-1]
        eigerr=float(np.linalg.norm(w-reference)/np.linalg.norm(reference))
        residual=float(np.linalg.norm(covariance@v-v*w)/np.linalg.norm(covariance))
        orth=float(np.linalg.norm(v.T@v-np.eye(d))/np.sqrt(d))
        assert eigerr < 3e-4 and residual < 3e-4 and orth < 3e-4,(n,d,eigerr,residual,orth)
        rows.append(dict(n=n,d=d,eigenvalue_error=eigerr,residual=residual,orthogonality=orth))
    return rows



def gram_cd():
    from mojolearn import ElasticNet
    rows=[]
    n,d=4096,16
    i=np.arange(n,dtype=np.uint32)
    x=np.empty((n,d),dtype=np.float32,order='F')
    for j in range(d):
        bits=i & np.uint32(j+1)
        parity=np.zeros(n,dtype=np.uint32)
        for b in range(12): parity ^= (bits >> np.uint32(b)) & np.uint32(1)
        x[:,j]=1-2*parity.astype(np.int32)
    beta=np.zeros(d,dtype=np.float32); beta[[0,1,5]]=[0.8,-0.6,0.2]
    y=x@beta
    assert not os.environ.get('MOJOLEARN_IDENTITY_TRACE'), 'trace would bypass Gram route'
    for ratio in (1.0,0.5):
        alpha=0.05
        fit=ElasticNet(alpha=alpha,l1_ratio=ratio,fit_intercept=False,max_iter=1000,tol=1e-7).fit(x,y)
        expected=np.sign(beta)*np.maximum(np.abs(beta)-alpha*ratio,0)/(1+alpha*(1-ratio))
        coef=np.asarray(fit.coef_,dtype=np.float64)
        error=float(np.max(np.abs(coef-expected)))
        assert error<3e-5,(ratio,error)
        rows.append(dict(n=n,d=d,l1_ratio=ratio,coefficient_max_error=error,route='untraced F-order public fit; no residual; cd_idn_gram_shape(4096,16)'))
    return rows


def small_eigh():
    from mojolearn import linalg
    rows=[]
    for n in (1,2,7,31,32,33):
        u=(1+(np.arange(n)%7)).astype(np.float32); u/=np.linalg.norm(u)
        x=np.eye(n,dtype=np.float32)*2+np.outer(u,u)
        values,vectors=linalg.eigh(x)
        w=np.asarray(values,dtype=np.float64);v=np.asarray(vectors,dtype=np.float64)
        residual=float(np.linalg.norm(x.astype(np.float64)@v-v*w)/np.linalg.norm(x))
        orth=float(np.linalg.norm(v.T@v-np.eye(n)))
        assert residual<2e-4 and orth<2e-4,(n,residual,orth)
        rows.append(dict(n=n,residual=residual,orthogonality=orth))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('gate',choices=('eigh','pca','gram_cd','small_eigh'))
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args()
    rows=globals()[a.gate]()
    from identical_wave_worker import source_provenance
    provenance=source_provenance(a.source.resolve(),os.environ['MOJOLEARN_VENDOR'])
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps({'status':'PASS','gate':a.gate,'checks':rows,'provenance':provenance,
        'scope':'quality only; does not establish cross-vendor identity or branch reachability'},indent=2)+'\n')
    print('IDENTICAL_WAVE_QUALITY',a.gate,'PASS',len(rows))
if __name__=='__main__':main()
