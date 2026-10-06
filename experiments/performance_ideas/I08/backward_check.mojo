# SPDX-License-Identifier: Apache-2.0
"""Full public Mamba2 backward consumer of retained forward stages.

Compares every gradient word across repeated real fits, emits all ten
complete-gradient digests for a frozen candidate/control pairing. This
fixture alone does not establish host/four-column identity or speed.
"""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from mamba.checks.mamba2_fixture import m2_case_weights,m2_case_x,m2_corpus_case
from mamba.impl.modules.mamba2_prefill_backward import mamba2_prefill_backward

def compare(name: String,a: List[Float32],b: List[Float32]) raises:
    if len(a)!=len(b):
        raise Error("I08 backward gradient shape differs")
    var hash=UInt64(1469598103934665603)
    for i in range(len(a)):
        var word=bitcast[DType.uint32](a[i])
        if word!=bitcast[DType.uint32](b[i]):
            raise Error("I08 repeated full backward differs: "+name)
        hash=(hash^UInt64(word))*UInt64(1099511628211)
    print("I08 BACKWARD_DIGEST",name,len(a),hash)

def main() raises:
    var ctx=DeviceContext()
    for case_k in [1,4]:
        var f=m2_corpus_case(case_k)
        var weights=m2_case_weights(case_k)
        var x=m2_case_x(case_k)
        var cotangent=List[Float32]()
        for i in range(len(x)):
            cotangent.append(Float32((i*37+11)%31-15)*Float32(0.0625))
        var a=mamba2_prefill_backward(weights,x,cotangent,f.b,f.l,f.dt_lo,f.dt_hi,ctx.copy())
        var b=mamba2_prefill_backward(weights,x,cotangent,f.b,f.l,f.dt_lo,f.dt_hi,ctx.copy())
        print("I08 BACKWARD_CASE",case_k,f.b,f.l,weights.dims.d_model)
        compare("x",a.x,b.x)
        compare("block_norm_weight",a.block_norm_weight,b.block_norm_weight)
        compare("in_proj_weight",a.in_proj_weight,b.in_proj_weight)
        compare("conv1d_weight",a.conv1d_weight,b.conv1d_weight)
        compare("conv1d_bias",a.conv1d_bias,b.conv1d_bias)
        compare("dt_bias",a.dt_bias,b.dt_bias)
        compare("A_log",a.A_log,b.A_log)
        compare("D",a.D,b.D)
        compare("norm_weight",a.norm_weight,b.norm_weight)
        compare("out_proj_weight",a.out_proj_weight,b.out_proj_weight)
    print("I08 BACKWARD_CONSUMER_PASS full_parameter_and_input_words short_cross_chunk_tail")
