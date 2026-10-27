#!/usr/bin/env python3
"""Bounded Apple ByteLM comparison. Legacy CPU is not qualified here."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import traceback


def run(output, baseline=None):
    import numpy as np
    from mojolearn import LanguageModelTrainer
    from mojolearn._byte_lm_config import ByteLanguageModelConfig
    from mojolearn import _byte_lm_impl
    from mojolearn._verify_native_runtime import audit_native_call
    shape=ByteLanguageModelConfig(batch=1,length=4,d_model=32,n_heads=4,n_kv=2,
                                  head_dim=8,intermediate=64,n_layers=1,vocab_size=32)
    rng=np.random.default_rng(79)
    params=(rng.standard_normal(shape.n_total)*.02).astype('f4')
    values, audits, counts={},[],{}
    def call(model, name, *args, **kwargs):
        if baseline:
            audit=audit_native_call(getattr(model,name),*args,**kwargs)
            audits.append(dict(operation=name,report=audit.report()))
            return audit.assert_native()
        return getattr(model,name)(*args,**kwargs)
    ids=np.array([[1,3,5]],dtype='i4')
    training=np.array([[1,3,5,7,9]],dtype='i4')
    for resident in (True,False):
        kwargs=dict(data_schedule={'batch_size':1,'vocab_size':32}, shape=shape,
                    resident=resident, step_result='lean' if resident else 'full')
        if baseline:
            audit=audit_native_call(LanguageModelTrainer,params,**kwargs)
            audits.append(dict(operation='constructor',report=audit.report()))
            model=audit.assert_native()
        else:
            model=LanguageModelTrainer(params,**kwargs)
        label='resident' if resident else 'stateless'
        held=call(model,'logits',ids)
        held_bytes=np.asarray(held).copy()
        values[label+'/initial']=held_bytes
        for repeat in range(2):
            result=call(model,'logits',ids)
            values[f'{label}/repeat{repeat}']=np.asarray(result).copy()
            assert np.asarray(held).tobytes()==held_bytes.tobytes()
            del result
            gc.collect()
        ragged=np.array([[1,3,999]],dtype='i4')
        values[label+'/ragged']=np.asarray(call(model,'logits',ragged,lengths=[2])).copy()
        assert not values[label+'/ragged'][:,2].view('u4').any()
        values[label+'/next']=np.asarray(call(model,'next_bytes',ids))
        values[label+'/ragged_next']=np.asarray(call(model,'next_bytes',ragged,lengths=[2]))
        if resident:
            for repeat in range(2):
                result=call(model,'train_step',training)
                for key in ('loss','step','flags'):
                    values[f'{label}/step{repeat}/{key}']=np.asarray(result[key]).copy()
            values[label+'/trained_logits']=np.asarray(call(model,'logits',ids)).copy()
        if baseline:
            counts[label]=_byte_lm_impl._LM_TRAINER(model,'native_provider_info',(),_byte_lm_impl._LM_TRAINER_CONTEXT)
            assert counts[label][0]=='mojolearn.training.byte_lm.v1'
            assert counts[label][1][2 if resident else 7] > 0
        try:
            model.logits(np.array([[shape.vocab_size]],dtype='i4'))
        except (ValueError,RuntimeError):
            pass
        else:
            raise AssertionError('out-of-range input did not refuse')
        values[label+'/after_refusal']=np.asarray(call(model,'logits',ids)).copy()
    if baseline:
        with np.load(baseline) as before:
            assert set(before.files)==set(values)
            for key,value in values.items():
                assert before[key].dtype==value.dtype and before[key].shape==value.shape,key
                assert before[key].tobytes()==value.tobytes(),key
        calls=[]
        class ForbiddenBuffer:
            def __buffer__(self, flags):
                calls.append(flags)
                raise AssertionError('Python buffer protocol reached')
        try:
            model.logits(ForbiddenBuffer())
        except TypeError:
            pass
        else:
            raise AssertionError('custom buffer was admitted')
        assert not calls
    np.savez(output.with_suffix('.npz'),**values)
    return dict(status='passed',cells=len(values),baseline=str(baseline) if baseline else None,
                arrays=str(output.with_suffix('.npz')),provider_counts=counts,runtime_audits=audits,
                sha256={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in values.items()},
                pending=['evaluate/checkpoints','stateless training','CPU binding','other targets','full measurement'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    args=parser.parse_args()
    try:
        report=run(args.output,args.baseline)
    except BaseException:
        report=dict(status='failed',traceback=traceback.format_exc())
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],evidence=str(args.output))))
    return report['status']!='passed'


if __name__=='__main__':
    raise SystemExit(main())
