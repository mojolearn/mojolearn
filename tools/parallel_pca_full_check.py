#!/usr/bin/env python3
"""Cloud-only full PCA through original distributed TSQR panels."""
import argparse
import hashlib
import json
import os
import struct
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cloud',action='store_true',required=True)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    if not os.environ.get('RUNPOD_POD_ID'):
        raise SystemExit('RunPod required; no local execution')
    import numpy as np
    from mojolearn import PCA
    from mojolearn.parallel_classical import fit_gram_estimator
    corpus=args.corpus.read_bytes()
    raw=np.frombuffer(corpus,dtype=np.uint8)
    checks=[]
    for rows,cols in ((17,7),(259,7),(259,17),(1025,65)):
        X=raw[:rows*cols].astype('<f4').reshape(rows,cols)/np.float32(255)
        for whiten in (False,True):
            opts=dict(n_components=3,svd_solver='full',whiten=whiten,numeric_mode='identical')
            one=PCA(**opts).fit(X)
            many=fit_gram_estimator(PCA(**opts),X,devices=(0,1))
            digest=hashlib.sha256()
            for name in ('components_','mean_','explained_variance_','explained_variance_ratio_','singular_values_'):
                a,b=getattr(one,name),getattr(many,name)
                assert a.shape==b.shape and a.tobytes()==b.tobytes(),(rows,cols,whiten,name)
                digest.update(b.tobytes())
            assert struct.pack('<d',one.noise_variance_)==struct.pack('<d',many.noise_variance_)
            a,b=one.transform(X),many.transform(X)
            assert a.tobytes()==b.tobytes()
            assert one.inverse_transform(a).tobytes()==many.inverse_transform(b).tobytes()
            before=many.components_.tobytes()
            try:
                fit_gram_estimator(many,X[:1],devices=(0,1))
            except (RuntimeError,ValueError):
                pass
            else:
                raise AssertionError('one-row full PCA accepted')
            assert many.components_.tobytes()==before
            checks.append(dict(rows=rows,features=cols,whiten=whiten,sha256=digest.hexdigest()))
            print('PASS full PCA',rows,cols,whiten,flush=True)
    args.report.write_text(json.dumps(dict(status='PASS',checks=checks,
        corpus_sha256=hashlib.sha256(corpus).hexdigest(),
        scope='Two H100s; original tall TSQR panels and final R fold; single-panel cases remain on root; no pooled root capacity'),indent=2)+'\n')


if __name__=='__main__':
    main()
