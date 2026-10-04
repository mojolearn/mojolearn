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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('gate',choices=('eigh','pca'))
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    rows=globals()[a.gate]()
    from identical_wave_worker import source_provenance
    provenance=source_provenance(Path(__file__).resolve().parents[1],os.environ['MOJOLEARN_VENDOR'])
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps({'status':'PASS','gate':a.gate,'checks':rows,'provenance':provenance,
        'scope':'quality only; does not establish cross-vendor identity or branch reachability'},indent=2)+'\n')
    print('IDENTICAL_WAVE_QUALITY',a.gate,'PASS',len(rows))
if __name__=='__main__':main()
