#!/usr/bin/env python3
"""Focused public GEMM provider probe; root runs it with the frozen Apple build."""
import argparse
import json
from pathlib import Path
import traceback


def run(output, baseline=None):
    import numpy as np
    from mojolearn import linalg
    from mojolearn import _linalg_impl as impl
    from mojolearn._verify_native_runtime import audit_native_call
    rng=np.random.default_rng(41)
    values, audits={},[]
    def call(name,*args,**kwargs):
        function=getattr(linalg,name)
        if baseline:
            audit=audit_native_call(function,*args,**kwargs)
            audits.append(dict(operation=name,report=audit.report()))
            return audit.assert_native()
        return function(*args,**kwargs)
    for m,n,k in ((7,5,8),(8,6,9)):
        a=rng.standard_normal((m,k)).astype('f4')
        b=rng.standard_normal((k,n)).astype('f4')
        label=f'{m}-{n}-{k}'
        for mode,aa,bb,kw in (
            ('nn',a,b,{}),('nt',a,b.T.copy(),{'transpose_b':True}),
            ('tn',a.T.copy(),b,{'transpose_a':True})):
            values[label+'/'+mode]=np.asarray(call('matmul',aa,bb,**kw)).copy()
        target=np.empty((m,n),dtype='f4')
        result=call('matmul',a,b,out=target)
        assert result is target
        values[label+'/out']=target.copy()
        ab=call('to_bf16',a)
        bb=call('to_bf16',b)
        values[label+'/bf16']=np.asarray(call('matmul_bf16',ab,bb)).copy()
        values[label+'/widen']=np.asarray(call('from_bf16',ab)).copy()
        qa=call('quantize_int8',a)
        qb=call('quantize_int8',b.T.copy())
        values[label+'/int8']=np.asarray(call('matmul_int8',qa,qb)).copy()
        values[label+'/dequant']=np.asarray(call('dequantize_int8',*qa)).copy()
    counts=None
    if baseline:
        counts=impl._NATIVE(None,'family_lin_provider_info',(None,),impl._NATIVE_CONTEXT)
        assert counts[0]=='mojolearn.linalg.v1'
        assert all(counts[1][op]>0 for op in (0,1,2,3,4,8,9))
        with np.load(baseline) as before:
            assert set(before.files)==set(values)
            for key,value in values.items():
                assert before[key].dtype==value.dtype and before[key].shape==value.shape,key
                assert before[key].tobytes()==value.tobytes(),key
    np.savez(output.with_suffix('.npz'),**values)
    return dict(status='passed',cells=len(values),provider_counts=counts,audits=audits,
                pending=['int15','CPU and older binding paths','other targets','full measurement'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        report=run(args.output,args.baseline)
    except Exception:
        args.output.write_text(json.dumps(dict(status='failed',error=traceback.format_exc()),indent=2)+'\n')
        raise
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],cells=report['cells'],report=str(args.output))))


if __name__=='__main__':
    main()
