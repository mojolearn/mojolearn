# SPDX-License-Identifier: Apache-2.0
"""NN20 public host-model adapter using the common scalar contract."""
from std.memory import bitcast
from core.host_lanes import host_f32_uninit
from checks.numerics import ftz, identical_exp, identical_div, identical_mul
from transformer.experiments.attention_summary_contract import (NN20_BALANCED_SUMMARY_TREE, summary_scratch_elements, summary_attention_forward_row, summary_attention_rowdot, summary_attention_dq_cell, summary_attention_dkdv_cell, _score, _dyv)

from transformer.experiments.summary_model_contract import (_require_model_shape, _host_absmax, _require_summary_regime, _require_summary_gradient_regime, _token_cell, _mask_row, _materialize_cell)

def model_summary_host_forward(q:List[Float32],k:List[Float32],v:List[Float32],
    b:Int,l:Int,nh:Int,nkv:Int,hd:Int,s:Int,pos0:Int,key_lo:Int,window:Int,scale:Float32,
    mut scores:List[Float32],mut masked:List[Float32],mut maxes:List[Float32],
    mut exps:List[Float32],mut denoms:List[Float32],mut probs:List[Float32],mut output:List[Float32]) raises:
    _require_model_shape(b,l,nh,nkv,hd,s,pos0,key_lo,window)
    var rows=b*nh*l
    var qpg=(nh//nkv)*l
    _require_summary_regime(_host_absmax(q),_host_absmax(k),_host_absmax(v),hd,s)
    var packed=host_f32_uninit(rows*hd)
    var result=host_f32_uninit(rows*hd)
    var scratch=host_f32_uninit(summary_scratch_elements(rows,s,hd,key_lo))
    var lo=List[Int32](length=rows,fill=Int32(0))
    var hi=List[Int32](length=rows,fill=Int32(0))
    var status=List[Int32](length=rows,fill=Int32(0))
    output=host_f32_uninit(rows*hd)
    for i in range(rows*hd):
        packed[i]=q[_token_cell(i,l,nh,hd)]
    var qp=rebind[MutPointer[Float32,MutAnyOrigin]](packed.unsafe_ptr())
    var kp=rebind[MutPointer[Float32,MutAnyOrigin]](k.unsafe_ptr())
    var vp=rebind[MutPointer[Float32,MutAnyOrigin]](v.unsafe_ptr())
    for row in range(rows):
        _mask_row(lo.unsafe_ptr(),hi.unsafe_ptr(),status.unsafe_ptr(),row,l,s,pos0,key_lo,window)
        summary_attention_forward_row[True](qp,kp,vp,lo.unsafe_ptr(),hi.unsafe_ptr(),result.unsafe_ptr(),maxes.unsafe_ptr(),denoms.unsafe_ptr(),scratch.unsafe_ptr(),status.unsafe_ptr(),row,s,hd,hd,qpg,scale,key_lo)
    for i in range(rows*hd):
        output[_token_cell(i,l,nh,hd)]=result[i]
    for cell in range(rows*s):
        _materialize_cell[False](qp,kp,vp,qp,lo.unsafe_ptr(),hi.unsafe_ptr(),maxes.unsafe_ptr(),denoms.unsafe_ptr(),maxes.unsafe_ptr(),scores.unsafe_ptr(),masked.unsafe_ptr(),exps.unsafe_ptr(),probs.unsafe_ptr(),cell,s,hd,qpg,scale)


def model_summary_host_backward(q:List[Float32],dy:List[Float32],k:List[Float32],v:List[Float32],
    maxes:List[Float32],denoms:List[Float32],b:Int,l:Int,nh:Int,nkv:Int,hd:Int,s:Int,pos0:Int,key_lo:Int,window:Int,scale:Float32,
    mut zdot:List[Float32],mut dq:List[Float32],mut dk:List[Float32],mut dv:List[Float32],
    mut dweights:List[Float32],mut dmasked:List[Float32],mut dscores:List[Float32],mut dqk:List[Float32]) raises:
    _require_model_shape(b,l,nh,nkv,hd,s,pos0,key_lo,window)
    var rows=b*nh*l
    var qpg=(nh//nkv)*l
    _require_summary_gradient_regime(_host_absmax(q),_host_absmax(k),_host_absmax(v),_host_absmax(dy),hd,s,qpg)
    var packed=host_f32_uninit(rows*hd)
    var pdy=host_f32_uninit(rows*hd)
    var pdq=host_f32_uninit(rows*hd)
    var lo=List[Int32](length=rows,fill=Int32(0))
    var hi=List[Int32](length=rows,fill=Int32(0))
    var status=List[Int32](length=rows,fill=Int32(0))
    zdot=host_f32_uninit(rows)
    dq=host_f32_uninit(rows*hd)
    dk=host_f32_uninit(b*nkv*s*hd)
    dv=host_f32_uninit(b*nkv*s*hd)
    dweights=host_f32_uninit(rows*s)
    dmasked=host_f32_uninit(rows*s)
    dscores=host_f32_uninit(rows*s)
    dqk=host_f32_uninit(rows*s)
    for i in range(rows*hd):
        packed[i]=q[_token_cell(i,l,nh,hd)]
        pdy[i]=dy[_token_cell(i,l,nh,hd)]
    var qp=rebind[MutPointer[Float32,MutAnyOrigin]](packed.unsafe_ptr())
    var kp=rebind[MutPointer[Float32,MutAnyOrigin]](k.unsafe_ptr())
    var vp=rebind[MutPointer[Float32,MutAnyOrigin]](v.unsafe_ptr())
    var mp=rebind[MutPointer[Float32,MutAnyOrigin]](maxes.unsafe_ptr())
    var zp=rebind[MutPointer[Float32,MutAnyOrigin]](denoms.unsafe_ptr())
    for row in range(rows):
        _mask_row(lo.unsafe_ptr(),hi.unsafe_ptr(),status.unsafe_ptr(),row,l,s,pos0,key_lo,window)
        summary_attention_rowdot(qp,kp,vp,pdy.unsafe_ptr(),lo.unsafe_ptr(),hi.unsafe_ptr(),mp,zp,zdot.unsafe_ptr(),status.unsafe_ptr(),row,s,hd,hd,qpg,scale)
    for i in range(rows*hd):
        summary_attention_dq_cell(qp,kp,vp,pdy.unsafe_ptr(),lo.unsafe_ptr(),hi.unsafe_ptr(),mp,zp,zdot.unsafe_ptr(),status.unsafe_ptr(),pdq.unsafe_ptr(),i,s,hd,hd,qpg,scale)
        dq[_token_cell(i,l,nh,hd)]=pdq[i]
    for i in range(b*nkv*s*hd):
        summary_attention_dkdv_cell(qp,kp,vp,pdy.unsafe_ptr(),lo.unsafe_ptr(),hi.unsafe_ptr(),mp,zp,zdot.unsafe_ptr(),status.unsafe_ptr(),dk.unsafe_ptr(),dv.unsafe_ptr(),i,rows,s,hd,hd,qpg,scale)
    for cell in range(rows*s):
        _materialize_cell[True](qp,kp,vp,pdy.unsafe_ptr(),lo.unsafe_ptr(),hi.unsafe_ptr(),mp,zp,zdot.unsafe_ptr(),dweights.unsafe_ptr(),dmasked.unsafe_ptr(),dscores.unsafe_ptr(),dqk.unsafe_ptr(),cell,s,hd,qpg,scale)
