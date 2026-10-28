#!/usr/bin/env python3
"""Public packed-storage probe, authored for root's combined frozen build."""
import argparse
import gc
import json
from pathlib import Path
import traceback


def run(output,baseline=None):
    import numpy as np
    from mojolearn import lowbit
    from mojolearn import _linalg_impl as impl
    from mojolearn._array import Array
    from mojolearn._verify_native_runtime import audit_native_call
    audits=[];values={}
    def call(fn,*args,**kw):
        if baseline:
            record=audit_native_call(fn,*args,**kw)
            audits.append({'operation':fn.__name__,'audit':record.report()})
            return record.assert_native()
        return fn(*args,**kw)
    x=np.arange(24,dtype='f4').reshape(4,6)/17
    for label,a in [('C',x),('F',np.asfortranarray(x)),('reverse',x[::-1,::-1])]:
        for fmt in ('float32','bfloat16','int8'):
            packed=call(lowbit.pack_one,a,fmt)
            plain=call(lowbit.materialize_one,packed)
            values[label+'/'+fmt]=np.asarray(plain).copy()
            if fmt!='float32':
                assert call(lowbit.is_packed,packed)
                assert call(lowbit.format_of,{'W':packed})==fmt
                copy=call(lowbit.pack,{'W':packed},fmt)['W']
                first=packed.bits if fmt=='bfloat16' else packed.codes
                second=copy.bits if fmt=='bfloat16' else copy.codes
                assert not np.shares_memory(np.asarray(first),np.asarray(second))
                assert np.asarray(first).tobytes()==np.asarray(second).tobytes()
                values[label+'/'+fmt+'/copy']=np.asarray(call(lowbit.materialize_one,copy)).copy()
                unpacked,kind=call(lowbit.unpack,{'W':packed,'bias':x[0]})
                assert kind==fmt and unpacked['bias'] is x[0]
                assert np.asarray(unpacked['W']).tobytes()==np.asarray(plain).tobytes()
    raw=np.arange(12,dtype='u2').reshape(3,4)
    packed=call(lowbit.BF16Weight,raw)
    assert isinstance(packed.bits,Array)
    assert np.shares_memory(np.asarray(packed.bits),raw)
    raw[0,0]=0x3f80
    assert np.asarray(packed.bits)[0,0]==0x3f80
    del raw;gc.collect()
    assert np.asarray(packed.bits)[0,0]==0x3f80
    q=np.arange(12,dtype='i1').reshape(3,4);e=np.arange(3,dtype='i4')
    packed8=call(lowbit.Int8Weight,q,e)
    assert isinstance(packed8.codes,Array) and isinstance(packed8.exponents,Array)
    assert np.shares_memory(np.asarray(packed8.codes),q)
    assert np.shares_memory(np.asarray(packed8.exponents),e)
    assert call(lowbit.format_of,[packed8])=='int8'
    assert call(lowbit.materialize_one,x) is x
    original={'W':x};result,kind=call(lowbit.unpack,original)
    assert result is original and kind=='float32'
    # Boundary conversion retains Array.from_list's floating conversion policy.
    nested=[[2**55+2**31+1,2**64+1],[1,-3]]
    values['nested']=np.asarray(call(lowbit.pack,{'W':nested},'float32')['W']).copy()
    values['f16']=np.asarray(call(impl.from_f16,np.array([0,0x3c00,0xc000],dtype='u2'))).copy()
    for fn,args,error in [
        (lowbit.BF16Weight,(x,),TypeError),
        (lowbit.Int8Weight,(q,np.zeros(4,dtype='i4')),TypeError),
        (lowbit.pack_one,(x.astype('f8'),'bfloat16'),TypeError),
        (lowbit.pack_one,(x,'bad'),ValueError),
        (lowbit.format_of,([packed,packed8],),ValueError),
    ]:
        try:call(fn,*args)
        except error:pass
        else:raise AssertionError((fn.__name__,'missing refusal'))
    if baseline:
        # Calling these Python API wrappers from a native operation is a regression.
        names=('to_bf16','from_bf16','quantize_int8','dequantize_int8','_load')
        saved={name:getattr(impl,name) for name in names}
        def poison(*args,**kw):raise AssertionError('Python linalg redispatch')
        try:
            for name in names:setattr(impl,name,poison)
            for fmt in ('bfloat16','int8'):
                call(lowbit.materialize_one,call(lowbit.pack_one,x,fmt))
        finally:
            for name,fn in saved.items():setattr(impl,name,fn)
        with np.load(baseline) as before:
            assert set(before.files)==set(values)
            for key,value in values.items():
                assert before[key].dtype==value.dtype and before[key].shape==value.shape,key
                assert before[key].tobytes()==value.tobytes(),key
    np.savez(output.with_suffix('.npz'),**values)
    return {'status':'passed','arrays':len(values),'audits':audits,'pending':['other vendor runtime','full measurements','remaining linalg/Array internals']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--baseline',type=Path);args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:report=run(args.output,args.baseline)
    except Exception:
        args.output.write_text(json.dumps({'status':'failed','error':traceback.format_exc()},indent=2)+'\n');raise
    args.output.write_text(json.dumps(report,indent=2)+'\n')
