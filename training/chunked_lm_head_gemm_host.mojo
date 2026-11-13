# SPDX-License-Identifier: Apache-2.0
"""ByteLM's explicit GEMM-logit chunked head, CPU forward and backward.

The separately exposed scalar-logit v2 API keeps its original oracle. This
model route uses selected neural GEMM logits, global ascending vocabulary
max/denominator/dHidden folds, ascending row dWeight folds and the existing
v2 mean-loss tree. A bounded panel is regenerated in each required pass;
no rows*vocab logits/dlogits owner is created. Source only, unverified.
"""
from std.memory import bitcast
from checks.numerics import ftz, identical_div, identical_exp, identical_fmax, identical_log, identical_mul_add
from gemm.contract import OP_NT
from gemm.host.neural_gemm import GhrPtr, gemm_host_rows_into
from training.checks.loss_contract import CE_NEG_INF_BITS, neg_by_bits, refuse_nonfinite
from training.checks.chunked_lm_head_oracle import (
    ChunkedLMHeadV2Result, CHUNKED_LM_HEAD_HOST_SABOTAGE, lm_head_v2_loss_fold,
)
from training.chunked_lm_head_profile import LM_HEAD_V2_CHUNK


def _admit(hidden: List[Float32], weight: List[Float32], targets: List[Int32],
           rows: Int, vocab: Int, width: Int) raises:
    if rows<1 or vocab<2 or width<1:
        raise Error("ByteLM chunked GEMM head requires positive rows/width and vocab >= 2")
    if rows>2147483647 or vocab>2147483647 or width>2147483647:
        raise Error("ByteLM chunked GEMM head dimension exceeds Int32")
    if rows*width>2147483647 or vocab*width>2147483647 or rows*min(vocab,LM_HEAD_V2_CHUNK)>2147483647:
        raise Error("ByteLM chunked GEMM head span exceeds Int32")
    if len(hidden)!=rows*width or len(weight)!=vocab*width or len(targets)!=rows:
        raise Error("ByteLM chunked GEMM head operand shape mismatch")
    refuse_nonfinite("ByteLM chunked head hidden",hidden)
    refuse_nonfinite("ByteLM chunked head weight",weight)
    for row in range(rows):
        if targets[row]<0 or Int(targets[row])>=vocab:
            raise Error("ByteLM chunked GEMM head target outside vocabulary")


def _panel(mut logits: List[Float32],hidden: List[Float32],weight: List[Float32],
           rows: Int,n: Int,width: Int,chunk0: Int) raises:
    gemm_host_rows_into(rebind[GhrPtr](hidden.unsafe_ptr()),
        rebind[GhrPtr](weight.unsafe_ptr()+chunk0*width),
        rebind[GhrPtr](logits.unsafe_ptr()),OP_NT,rows,n,width)


def byte_chunked_head_forward(hidden: List[Float32],weight: List[Float32],
    targets: List[Int32],rows: Int,vocab: Int,width: Int) raises -> ChunkedLMHeadV2Result:
    _admit(hidden,weight,targets,rows,vocab,width)
    var panel = List[Float32](length=rows*min(vocab,LM_HEAD_V2_CHUNK),fill=Float32(0))
    var maxima = List[Float32](length=rows,fill=bitcast[DType.float32](CE_NEG_INF_BITS))
    var denom = List[Float32](length=rows,fill=Float32(0))
    var row_loss = List[Float32](length=rows,fill=Float32(0))
    for chunk0 in range(0,vocab,LM_HEAD_V2_CHUNK):
        var n = min(LM_HEAD_V2_CHUNK,vocab-chunk0)
        _panel(panel,hidden,weight,rows,n,width,chunk0)
        for row in range(rows):
            var acc = maxima[row]
            for j in range(n):
                acc = identical_fmax(acc,panel[row*n+j])
            maxima[row] = acc
    for chunk0 in range(0,vocab,LM_HEAD_V2_CHUNK):
        var n = min(LM_HEAD_V2_CHUNK,vocab-chunk0)
        _panel(panel,hidden,weight,rows,n,width,chunk0)
        for row in range(rows):
            var acc = denom[row]
            for j in range(n):
                var shifted = ftz(ftz(panel[row*n+j])-ftz(maxima[row]))
                acc = ftz(acc+ftz(identical_exp(shifted)))
                if chunk0+j==Int(targets[row]):
                    row_loss[row] = shifted
            denom[row] = acc
    for row in range(rows):
        row_loss[row] = neg_by_bits(ftz(ftz(row_loss[row])-ftz(identical_log(ftz(denom[row])))))
    var loss = lm_head_v2_loss_fold(row_loss,rows)
    refuse_nonfinite("ByteLM chunked head maxima",maxima)
    refuse_nonfinite("ByteLM chunked head denominator",denom)
    refuse_nonfinite("ByteLM chunked head row loss",row_loss)
    return ChunkedLMHeadV2Result(loss,maxima^,denom^,List[Float32](),List[Float32]())


def byte_chunked_head_backward(hidden: List[Float32],weight: List[Float32],
    targets: List[Int32],rows: Int,vocab: Int,width: Int) raises -> ChunkedLMHeadV2Result:
    var fwd = byte_chunked_head_forward(hidden,weight,targets,rows,vocab,width)
    var panel = List[Float32](length=rows*min(vocab,LM_HEAD_V2_CHUNK),fill=Float32(0))
    var dh = List[Float32](length=rows*width,fill=Float32(0))
    var dw = List[Float32](length=vocab*width,fill=Float32(0))
    for chunk0 in range(0,vocab,LM_HEAD_V2_CHUNK):
        var n = min(LM_HEAD_V2_CHUNK,vocab-chunk0)
        _panel(panel,hidden,weight,rows,n,width,chunk0)
        # Each hidden cell carries its exact ascending vocabulary chain
        # through panel boundaries, including each stored FMA result.
        for row in range(rows):
            for j in range(n):
                var shifted = ftz(ftz(panel[row*n+j])-ftz(fwd.row_max[row]))
                var probability = ftz(identical_div(ftz(identical_exp(shifted)),ftz(fwd.row_denom[row])))
                var target = Float32(1) if chunk0+j==Int(targets[row]) else Float32(0)
                var dl = ftz(identical_div(ftz(probability-target),Float32(rows)))
                for feature in range(width):
                    var cell = row*width+feature
                    dh[cell] = identical_mul_add(dl,weight[(chunk0+j)*width+feature],dh[cell])
        for j in range(n):
            for r in range(rows):
                var row = r
                comptime if CHUNKED_LM_HEAD_HOST_SABOTAGE:
                    row = rows-1-r
                var shifted = ftz(ftz(panel[row*n+j])-ftz(fwd.row_max[row]))
                var probability = ftz(identical_div(ftz(identical_exp(shifted)),ftz(fwd.row_denom[row])))
                var target = Float32(1) if chunk0+j==Int(targets[row]) else Float32(0)
                var dl = ftz(identical_div(ftz(probability-target),Float32(rows)))
                for feature in range(width):
                    var cell = (chunk0+j)*width+feature
                    dw[cell] = identical_mul_add(dl,hidden[row*width+feature],dw[cell])
            for feature in range(width):
                var cell = (chunk0+j)*width+feature
                dw[cell] = ftz(dw[cell])
    refuse_nonfinite("ByteLM chunked head dHidden",dh)
    refuse_nonfinite("ByteLM chunked head dWeight",dw)
    var maxima = fwd.row_max.copy()
    var denom = fwd.row_denom.copy()
    return ChunkedLMHeadV2Result(fwd.loss,maxima^,denom^,dh^,dw^)
