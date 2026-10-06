#!/usr/bin/env python3
# F06 PENDING: new variants are FAST Apple/default off; incumbent paths retained.
"""Independent batching, active compaction and exact-support covariance reuse.

NumPy generates fixtures and independent quality references only. Every
measured fit, distance, prediction and score calls the actual product API.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apple_fast'))
from support import capture_main,binding_check,consumed


def exercise(args):
    import numpy as np
    import resource
    from mojolearn import MinCovDet,EllipticEnvelope
    from mojolearn import _mojolearn_x_decomp as binding
    cases={}
    reuse_total=0
    for n,d,contamination,singular in ((313,5,.1,False),(601,9,.2,False),(907,13,.1,False),(383,7,.15,True)):
        rng=np.random.default_rng(514)
        clean=rng.normal(size=(n,d)).astype('float32')
        clean[:,-1]=clean[:,0]+np.float32(0 if singular else .003)*clean[:,-1]
        data=clean.copy();data[:int(n*contamination)]+=np.float32(8)
        query=rng.normal(size=(97,d)).astype('float32')
        query[:,-1]=query[:,0]+np.float32(0 if singular else .003)*query[:,-1]
        labels=np.ones(97,dtype=np.int32);labels[:17]=-1
        query[:17]+=np.float32(8)
        for kind in ('mcd','elliptic'):
            model=MinCovDet(random_state=7,numeric_mode='fast') if kind=='mcd' else EllipticEnvelope(random_state=7,contamination=contamination,numeric_mode='fast')
            before=int(binding.mcd_fast_batch_count())
            stages=[int(binding.mcd_experiment_count(i)) for i in range(3)]
            def fit():
                model.fit(data)
                return np.asarray(model.location_),np.asarray(model.covariance_),np.asarray(model.support_)
            _,fit_ms=consumed(fit)
            reached=int(binding.mcd_fast_batch_count())-before
            hits=[int(binding.mcd_experiment_count(i))-stages[i] for i in range(3)]
            reuse_total+=hits[2]
            if args.arm=='B':
                if args.variant=='default' and reached==0:raise AssertionError('bounded MCD batches never reached')
                if args.variant in ('active_compact','active_cov') and hits[0]==0:raise AssertionError('compact MCD product entrance never reached')
                if args.variant in ('cov_reuse','active_cov') and hits[1]==0:raise AssertionError('phase-local covariance reuse entrance never reached')
            elif any(hits):raise AssertionError('default-off MCD experiment reached in baseline')
            loc=np.asarray(model.location_,float);cov=np.asarray(model.covariance_,float)
            oracle=np.cov(clean[int(n*contamination):].astype(float),rowvar=False)
            covariance_error=float(np.linalg.norm(cov-oracle)/np.linalg.norm(oracle))
            location_error=float(np.linalg.norm(loc))
            distance,mahalanobis_ms=consumed(lambda: np.asarray(model.mahalanobis(query)))
            distance=np.asarray(distance,dtype=float)
            if distance.shape!=(97,) or not np.isfinite(distance).all() or (distance<0).any():raise AssertionError('invalid actual Mahalanobis consumer')
            # Contamination discrimination independent of fitted matrix bits.
            auc=float((distance[:17,None]>distance[None,17:]).mean())
            metrics=dict(covariance=dict(value=covariance_error,rtol=.01,atol=1e-5),
                location=dict(value=location_error,rtol=.01,atol=1e-5),
                contamination_ranking_error=dict(value=1-auc,rtol=.01,atol=1/1360))
            consumer_ms=dict(mahalanobis=mahalanobis_ms)
            score_classification='finite'
            if kind=='elliptic':
                predicted,predict_ms=consumed(lambda: np.asarray(model.predict(query)))
                decision,decision_ms=consumed(lambda: np.asarray(model.decision_function(query)))
                scores,score_samples_ms=consumed(lambda: np.asarray(model.score_samples(query)))
                accuracy,score_ms=consumed(lambda: float(model.score(query,labels)))
                decision=np.asarray(decision,float);scores=np.asarray(scores,float)
                if not np.isfinite(decision).all() or not np.isfinite(scores).all():raise AssertionError('nonfinite actual elliptic score')
                if not np.array_equal(np.asarray(predicted),np.where(decision>=0,1,-1)):raise AssertionError('predict/decision consumers disagree')
                metrics['classification_error']=dict(value=1-float(accuracy),rtol=.01,atol=1/97)
                consumer_ms.update(predict=predict_ms,decision_function=decision_ms,score_samples=score_samples_ms,score=score_ms)
            else:
                likelihood,score_ms=consumed(lambda: float(model.score(query)))
                if np.isnan(likelihood) or np.isposinf(likelihood):raise AssertionError('invalid actual covariance score')
                score_classification='negative_infinity' if np.isneginf(likelihood) else 'finite'
                consumer_ms['score']=score_ms
            cases[f'{kind}-{n}-{d}-singular{int(singular)}']=dict(contract=dict(n=n,d=d,contamination=contamination,seed=7,singular=singular,query_rows=97,score_classification=score_classification),
                metrics=metrics,fit_ms=fit_ms,consumer_ms=consumer_ms,batch_launches=reached,
                compact_submissions=hits[0],reuse_preparations=hits[1],reused_candidates=hits[2],
                support_count=int(np.asarray(model.support_).sum()),peak_process_rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if args.arm=='B' and args.variant in ('cov_reuse','active_cov') and reuse_total==0:raise AssertionError('no exact-support covariance reuse in actual fits')
    return dict(binding=binding_check(binding,'x_decomp'),cases=cases)
if __name__=='__main__':capture_main(exercise)
