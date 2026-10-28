#!/usr/bin/env python3
"""Authored CPU public probe; root schedules it after the combined compile."""
import argparse
import json
from pathlib import Path
import traceback


def run(path):
    import numpy as np
    from mojolearn import linalg
    from mojolearn import _linalg_impl as impl
    from mojolearn._verify_native_runtime import audit_native_call
    binding=impl._load()
    assert binding.__name__.endswith('_host'),binding.__name__
    values={};audits=[]
    def call(name,*args,**kwargs):
        record=audit_native_call(getattr(linalg,name),*args,**kwargs)
        audits.append({'operation':name,'audit':record.report()})
        return record.assert_native()
    a=np.arange(24,dtype='f4').reshape(4,6)/17
    b=np.arange(30,dtype='f4').reshape(5,6)/13
    values['gemm']=np.asarray(call('matmul',a,b,transpose_b=True)).copy()
    bits=call('to_bf16',a)
    values['bf16']=np.asarray(call('from_bf16',bits)).copy()
    for width in (8,15):
        qa=call('quantize_int'+str(width),a)
        qb=call('quantize_int'+str(width),b)
        values['int'+str(width)]=np.asarray(call('matmul_int'+str(width),qa,qb)).copy()
        values['dequant'+str(width)]=np.asarray(call('dequantize_int'+str(width),*qa)).copy()
    values['matmul_bf16']=np.asarray(call('matmul_bf16',bits,call('to_bf16',b),transpose_b=True)).copy()
    # Once connected, old Python-callable numerical doors must not drive work.
    names=('gemm','gemm_bf16','gemm_int8','gemm_int15','quantize_int8','dequantize_int8','quantize_int15','dequantize_int15','to_bf16','from_bf16')
    saved={name:getattr(binding,name) for name in names}
    def poison(*args):
        raise AssertionError('legacy numerical dispatch')
    try:
        for name in names:setattr(binding,name,poison)
        assert np.array_equal(np.asarray(call('matmul',a,b,transpose_b=True)),values['gemm'])
        call('from_bf16',call('to_bf16',a))
        call('dequantize_int8',*call('quantize_int8',a))
        call('dequantize_int15',*call('quantize_int15',a))
    finally:
        for name,value in saved.items():setattr(binding,name,value)
    np.savez(path.with_suffix('.npz'),**values)
    return {'status':'passed','binding':binding.__name__,'arrays':len(values),'audits':audits,'pending':['from_f16 public surface','other algorithm families','measurements']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:report=run(args.output)
    except Exception:
        args.output.write_text(json.dumps({'status':'failed','error':traceback.format_exc()},indent=2)+'\n');raise
    args.output.write_text(json.dumps(report,indent=2)+'\n')
