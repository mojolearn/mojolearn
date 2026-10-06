#!/usr/bin/env python3
"""Actual TSQR least-squares decomposition on rank and conditioning fixtures."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main,binding_check,consumed

def exercise(args):
    import numpy as np
    if args.variant == "compensated-pca":
        return pca(args)
    from mojolearn import lstsq, randomized_svd
    from mojolearn import _mojolearn_x_decomp as binding
    cases={}
    for rows,columns,condition in ((509,17,1e2),(521,19,1e4),(997,31,1e6)):
        rng=np.random.default_rng(981)
        u,_=np.linalg.qr(rng.normal(size=(rows,columns)))
        v,_=np.linalg.qr(rng.normal(size=(columns,columns)))
        s=np.geomspace(1,1/condition,columns)
        a=np.asarray((u*s)@v.T,'float32');truth=rng.normal(size=(columns,3));b=np.asarray(a.astype(float)@truth,'float32')
        result,elapsed=consumed(lambda:lstsq(a,b,rcond=None,numeric_mode='fast'))
        answer=np.asarray(result[0],float)
        residual=float(np.linalg.norm(a.astype(float)@answer-b)/max(np.linalg.norm(b),1e-9))
        state_error=float(np.linalg.norm(answer-truth)/np.linalg.norm(truth))
        cases[f'{rows}-{columns}-{condition}']=dict(contract=dict(rows=rows,columns=columns,condition=condition,seed=981),
            metrics=dict(residual=dict(value=residual,rtol=.1,atol=2e-7),coefficient_error=dict(value=state_error,rtol=.01,atol=1e-5)),
            fit_ms=elapsed,rank=int(result[2]),singular_values=np.asarray(result[3]).tolist())
        # Public randomized range-finder preserves its requested rank/seed and
        # checks the complete reconstructed result against independent SVD.
        rank = min(7, columns-1)
        result, range_ms = consumed(lambda: randomized_svd(a, rank, n_oversamples=5, n_iter=3, random_state=981, numeric_mode='fast'))
        ur, sr, vr = result
        reconstruction = np.asarray(ur,float) @ (np.asarray(sr,float)[:,None]*np.asarray(vr,float))
        exact = np.linalg.svd(a.astype(float),compute_uv=False)
        optimal = float(np.linalg.norm(exact[rank:])/np.linalg.norm(exact))
        range_error = float(np.linalg.norm(a.astype(float)-reconstruction)/np.linalg.norm(a))
        cases[f'range-{rows}-{columns}-{condition}'] = dict(
            contract=dict(rows=rows,columns=columns,condition=condition,rank=rank,seed=981,iterations=3),
            metrics=dict(excess_reconstruction=dict(value=max(0,range_error-optimal),rtol=.1,atol=2e-6)),range_ms=range_ms)
    return dict(binding=binding_check(binding,'x_decomp'),cases=cases)
def pca(args):
    import numpy as np
    import time
    from mojolearn import PCA,LocallyLinearEmbedding
    from mojolearn import _mojolearn_estimators as binding
    cases={}
    for kind in ('collinear','low-rank','clustered'):
        rng=np.random.default_rng(981)
        rows,cols,rank=997,(137 if args.variant=="pca" else 37),7
        u,_=np.linalg.qr(rng.normal(size=(rows,cols)))
        v,_=np.linalg.qr(rng.normal(size=(cols,cols)))
        spectrum=np.geomspace(1,.0001,cols) if kind=='collinear' else (np.r_[np.linspace(1,.8,rank),np.full(cols-rank,1e-6)] if kind=='low-rank' else np.r_[np.ones(rank),np.full(cols-rank,.5)])
        a=np.asarray((u*spectrum)@v.T+np.linspace(-2,2,cols),'float32')
        saved=a.copy(); centered=a.astype(float)-a.astype(float).mean(axis=0)
        truth=np.linalg.svd(centered,compute_uv=False)
        before=int(binding.pca_compensated_cov_count())
        start=time.perf_counter_ns();model=PCA(n_components=rank,numeric_mode='fast').fit(a)
        singular=np.asarray(model.singular_values_,float);singular.tobytes()
        fit_ms=(time.perf_counter_ns()-start)/1e6
        reached=int(binding.pca_compensated_cov_count())-before
        if args.arm=="B": assert reached>0,"selective compensated covariance never reached"
        assert np.array_equal(a,saved),'PCA changed caller input'
        projected,transform_ms=consumed(lambda:model.transform(a))
        reconstructed,inverse_ms=consumed(lambda:model.inverse_transform(projected))
        recon=float(np.linalg.norm(centered-(np.asarray(reconstructed,float)-a.astype(float).mean(axis=0)))/np.linalg.norm(centered))
        optimal=float(np.linalg.norm(truth[rank:])/np.linalg.norm(truth))
        singular_error=float(np.linalg.norm(singular-truth[:rank])/np.linalg.norm(truth[:rank]))
        noise_truth=float(np.sum(truth[rank:]**2)/((rows-1)*(cols-rank)))
        noise_error=abs(float(model.noise_variance_)-noise_truth)
        # Downstream embedding uses the same requested neighbors/dimensions.
        # Do not turn a PCA transform gain into permission to change fit.
        downstream=LocallyLinearEmbedding(n_neighbors=11,n_components=3,method='standard',numeric_mode='fast')
        embedding_input=np.asarray(projected)[:127]
        embedding,embedding_ms=consumed(lambda:downstream.fit_transform(embedding_input))
        original_dist=np.sum((np.asarray(embedding_input,float)[:,None,:]-np.asarray(embedding_input,float)[None,:,:])**2,axis=2)
        embedded_dist=np.sum((np.asarray(embedding,float)[:,None,:]-np.asarray(embedding,float)[None,:,:])**2,axis=2)
        original_neigh=np.argsort(original_dist,axis=1)[:,1:12]
        embedded_neigh=np.argsort(embedded_dist,axis=1)[:,1:12]
        overlap=sum(len(set(a)&set(b)) for a,b in zip(original_neigh,embedded_neigh))/(len(embedding_input)*11)
        embedding_error=float(1-overlap)
        cases[kind]=dict(contract=dict(rows=rows,cols=cols,rank=rank,seed=981,kind=kind,embedding_rows=127),
            metrics=dict(excess_reconstruction=dict(value=max(0,recon-optimal),rtol=.1,atol=2e-6),
                singular_error=dict(value=singular_error,rtol=.1,atol=2e-6),noise_error=dict(value=noise_error,rtol=.1,atol=1e-9),embedding_error=dict(value=embedding_error,rtol=0,atol=.005)),
            fit_ms=fit_ms,transform_ms=transform_ms,inverse_ms=inverse_ms,embedding_ms=embedding_ms,compensated_products=reached)
    return dict(binding=binding_check(binding,'estimators'),cases=cases)
if __name__=='__main__':capture_main(exercise)
