# SPDX-License-Identifier: Apache-2.0
"""Shared scalar attention layout, admission and trace operations; no GPU imports."""
from std.memory import bitcast
from checks.numerics import ftz, identical_exp, identical_div, identical_mul
from transformer.experiments.attention_summary_contract import (NN20_BALANCED_SUMMARY_TREE, summary_scratch_elements, summary_attention_forward_row, summary_attention_rowdot, summary_attention_dq_cell, summary_attention_dkdv_cell, _score, _dyv)

def regime_finite(v: Float64) -> Bool:
    return v < Float64(bitcast[DType.float32](UInt32(0x7F800000)))

def regime_product_ok(n: Int, a: Float64, b: Float64) -> Bool:
    return regime_finite(a) and regime_finite(b) and Float64(n)*a*b < Float64(1267650600228229401496703205376.0)

def _require_model_shape(b:Int,l:Int,nh:Int,nkv:Int,hd:Int,s:Int,
                         pos0:Int,key_lo:Int,window:Int) raises:
    # These are public tensor/index ABI bounds, not performance thresholds.
    # Every generated mask is then 0 <= lo < hi <= s; its status is zero
    # by construction, with no device-to-host status scan needed per row.
    if b <= 0 or l <= 0 or nh <= 0 or nkv <= 0 or hd <= 0 or s <= 0:
        raise Error("NN20: positive model dimensions required")
    if nh % nkv != 0 or key_lo < 0 or pos0 < key_lo or s != pos0+l-key_lo or window < 0:
        raise Error("NN20: invalid grouped-head or absolute key-span metadata")
    if max(max(b*nh*l*hd,b*nh*l*s),b*nkv*s*hd) > 2147483647:
        raise Error("NN20: model linear index exceeds Int32 kernel ABI")
    if pos0+l > 2147483647 or window > 2147483647:
        raise Error("NN20: absolute position exceeds Int32 kernel ABI")


def _host_absmax(x: List[Float32]) -> Float64:
    var maximum = Float64(0.0)
    for i in range(len(x)):
        var bits = bitcast[DType.uint32](x[i]) & UInt32(0x7FFFFFFF)
        if bits >= UInt32(0x7F800000):
            return Float64(bitcast[DType.float32](UInt32(0x7F800000)))
        maximum = max(maximum, Float64(bitcast[DType.float32](bits)))
    return maximum


def _require_summary_regime(qm: Float64, km: Float64, vm: Float64, hd: Int, s: Int) raises:
    if not regime_product_ok(hd,qm,km) or not regime_product_ok(s,vm,Float64(1.0)):
        raise Error("NN20: nonfinite or overflow-risk attention operands")


def _require_summary_gradient_regime(qm:Float64,km:Float64,vm:Float64,dm:Float64,hd:Int,s:Int,qpg:Int) raises:
    _require_summary_regime(qm,km,vm,hd,s)
    if not regime_finite(dm):
        raise Error("NN20: nonfinite or overflow-risk attention gradient")
    var dyv = Float64(hd)*dm*vm
    if (not regime_product_ok(s,Float64(2.0)*dyv,km)
            or not regime_product_ok(qpg,Float64(2.0)*dyv,qm)
            or not regime_product_ok(qpg,dm,Float64(1.0))):
        raise Error("NN20: nonfinite or overflow-risk attention gradient")


def _token_cell(i: Int, l: Int, nh: Int, hd: Int) -> Int:
    var d = i % hd
    var row = i // hd
    var t = row % l
    var h = (row // l) % nh
    var bb = row // (l * nh)
    return (bb * l + t) * nh * hd + h * hd + d


def _mask_row(lo: MutPointer[Int32, MutAnyOrigin], hi: MutPointer[Int32, MutAnyOrigin],
              status: MutPointer[Int32, MutAnyOrigin], row: Int, l: Int, s: Int,
              pos0: Int, key_lo: Int, window: Int):
    var p = pos0 + row % l
    var begin = 0
    if window > 0:
        begin = max(0, p - window + 1 - key_lo)
    lo.unsafe_store(row, Int32(begin))
    hi.unsafe_store(row, Int32(min(s, p - key_lo + 1)))
    status.unsafe_store(row, Int32(0))


def _materialize_cell[BACKWARD: Bool](q: MutPointer[Float32, MutAnyOrigin],
    k: MutPointer[Float32, MutAnyOrigin], v: MutPointer[Float32, MutAnyOrigin],
    dy: MutPointer[Float32, MutAnyOrigin], lo: MutPointer[Int32, MutAnyOrigin],
    hi: MutPointer[Int32, MutAnyOrigin], maxes: MutPointer[Float32, MutAnyOrigin],
    denoms: MutPointer[Float32, MutAnyOrigin], zdot: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin], b: MutPointer[Float32, MutAnyOrigin],
    c: MutPointer[Float32, MutAnyOrigin], d: MutPointer[Float32, MutAnyOrigin],
    cell: Int, s: Int, hd: Int, qpg: Int, scale: Float32):
    var row = cell // s
    var j = cell % s
    var group = row // qpg
    var visible = j >= Int(lo.unsafe_load(row)) and j < Int(hi.unsafe_load(row))
    var score = _score(q,k,row,group,j,s,hd,scale)
    var e = Float32(0.0)
    var p = Float32(0.0)
    if visible:
        e = ftz(identical_exp(ftz(score-maxes.unsafe_load(row))))
        p = ftz(identical_div(e,denoms.unsafe_load(row)))
    comptime if BACKWARD:
        var gy = _dyv(v,dy,row,group,j,s,hd)
        var ds = Float32(0.0)
        if visible:
            ds = ftz(identical_mul(p,ftz(gy-zdot.unsafe_load(row))))
        a.unsafe_store(cell,gy)
        b.unsafe_store(cell,ds)
        c.unsafe_store(cell,ds)
        d.unsafe_store(cell,ftz(identical_mul(ds,scale)))
    else:
        a.unsafe_store(cell,score)
        b.unsafe_store(cell,score if visible else bitcast[DType.float32](UInt32(0xFF800000)))
        c.unsafe_store(cell,e)
        d.unsafe_store(cell,p)


