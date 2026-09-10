#!/usr/bin/env python3
"""Large NVIDIA RF inference: same forest sequential/parallel groves, cuML context."""
import argparse, hashlib, json, os, statistics, time
from pathlib import Path
import numpy as np
import cupy
import forest_speed_arm as forest
from nvidia_identical_trees import digest_arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows',type=int,default=1000000)
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.rows<2 or args.rounds<1: parser.error('positive rounds and rows>=2 required')
    if os.environ.get('MOJOLEARN_NUMERIC_MODE')!='identical': parser.error('select IDENTICAL')
    if args.output.exists(): parser.error('choose fresh output to preserve evidence')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    data=forest.spec.load_dataset('higgs','shipped',args.rows)
    forest.spec.emit_scale_reminder(data,'inference-before')
    forest.prepare_our_inputs(data);forest.spec.prepare_cuml_labels(data)
    cfg=forest.spec.lane_config('rf','shipped')
    arm=forest.our_rf_arm('rf',cfg,data);witness=forest.verify_our_arm(arm)
    assert witness['vendor']=='cuda'
    ours=arm.make();arm.fit(ours,data);arm.sync()
    competitor=forest.spec.cuml_rf_arm('rf',cfg,data)
    cuml_model=competitor.make();competitor.fit(cuml_model,data);competitor.sync()
    hashes={}; reference=None; records=[]
    for round_index in range(args.rounds+1):
        names=('sequential','parallel_groves','cuml_gpu')
        offset=max(0,round_index-1)%3
        names=names[offset:]+names[:offset]
        for name in names:
            if name!='cuml_gpu': ours.inference_engine=name
            start=time.perf_counter()
            if name=='cuml_gpu':
                pred=cuml_model.predict_proba(data.X_test)
                if isinstance(pred,cupy.ndarray): pred=cupy.asnumpy(pred)
                pred=np.asarray(pred)
                cupy.cuda.runtime.deviceSynchronize()
            else:
                pred=ours.predict_proba(data._ours_Xtest)
            ms=1000*(time.perf_counter()-start)
            assert pred.shape==(len(data.y_test),2) and np.isfinite(pred).all()
            digest=digest_arrays([('probabilities',pred)])
            if name in hashes and name!='cuml_gpu': assert digest==hashes[name]
            hashes[name]=digest
            if name=='sequential': reference=pred.copy()
            if name=='parallel_groves': np.testing.assert_allclose(pred,reference,rtol=2e-6,atol=2e-6)
            row=dict(arm=name,round=round_index,warmup=round_index==0,ms=ms,hash=digest)
            records.append(row)
            print(json.dumps(row),flush=True)
            args.output.write_text(json.dumps(dict(records=records,complete=False),indent=2)+'\n')
    summary={}
    for name in hashes:
        samples=[r['ms'] for r in records if r['arm']==name and not r['warmup']]
        summary[name]=dict(samples_ms=samples,median_ms=statistics.median(samples),spread=max(samples)/min(samples),stable=len(samples)>=5 and max(samples)/min(samples)<=1.10)
    result=dict(records=records,summary=summary,complete=True,rows_fit=len(data.y_train),rows_predict=len(data.y_test),features=data.X_train.shape[1],numeric_mode='identical',vendor='cuda',binding=witness,config=cfg,data_sha256=digest_arrays([('X_test',data.X_test)]),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),contract='public predict_proba includes host input/output transfers; fit excluded; same MojoLearn forest; cuML uses its separately trained cached GPU model; different summation algorithms may differ in bits')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    forest.spec.emit_scale_reminder(data,'inference-after')
    print('PASS large RF inference comparison')

if __name__=='__main__':main()
