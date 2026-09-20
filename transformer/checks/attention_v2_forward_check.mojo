# SPDX-License-Identifier: Apache-2.0
"""Apple gate for opt-in TILE=32 attention-v2 forward.

Expected words come from tools/attention_v2_oracle.py.  Rows cover a 65-key
tail, a nonzero sliding window, a one-cell causal mask, and exponent extremes.
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceContext
from transformer.impl.llama.attention_v2 import enqueue_attention_v2_forward


def main() raises:
    comptime assert is_defined["MOJOLEARN_NUMERIC_IDENTICAL"]()
    comptime R = 4
    comptime K = 65
    comptime D = 3
    var ctx = DeviceContext()
    var hs = ctx.enqueue_create_host_buffer[DType.float32](R * K)
    var hv = ctx.enqueue_create_host_buffer[DType.float32](R * K * D)
    for r in range(R):
        for j in range(K):
            hs[r * K + j] = Float32(((j * 17 + r * 13) % 29 - 14)) / Float32(4.0)
            for d in range(D):
                hv[(r * K + j) * D + d] = Float32(((j * 7 + d * 11 + r * 5) % 31 - 15)) / Float32(8.0)
    for j in range(K):
        hs[3 * K + j] = Float32(-80.0)
    hs[3 * K + 31] = Float32(0.0)
    hs[3 * K + 64] = Float32(80.0)
    var hlo = ctx.enqueue_create_host_buffer[DType.int32](R)
    var hhi = ctx.enqueue_create_host_buffer[DType.int32](R)
    hlo[0] = 0; hhi[0] = 65
    hlo[1] = 7; hhi[1] = 40
    hlo[2] = 0; hhi[2] = 1
    hlo[3] = 0; hhi[3] = 65
    var dq = ctx.enqueue_create_buffer[DType.float32](R)
    var hq = ctx.enqueue_create_host_buffer[DType.float32](R)
    for r in range(R): hq[r] = Float32(1.0)
    var dk = ctx.enqueue_create_buffer[DType.float32](R * K)
    var dv = ctx.enqueue_create_buffer[DType.float32](R * K * D)
    var dlo = ctx.enqueue_create_buffer[DType.int32](R)
    var dhi = ctx.enqueue_create_buffer[DType.int32](R)
    var dout = ctx.enqueue_create_buffer[DType.float32](R * D)
    var dm = ctx.enqueue_create_buffer[DType.float32](R)
    var dz = ctx.enqueue_create_buffer[DType.float32](R)
    ctx.enqueue_copy(dst_buf=dq, src_buf=hq)
    ctx.enqueue_copy(dst_buf=dk, src_buf=hs)
    ctx.enqueue_copy(dst_buf=dv, src_buf=hv)
    ctx.enqueue_copy(dst_buf=dlo, src_buf=hlo)
    ctx.enqueue_copy(dst_buf=dhi, src_buf=hhi)
    enqueue_attention_v2_forward(ctx, dq, dk, dv, dlo, dhi, dout, dm, dz, R, K, D, 1, 1, Float32(1.0))
    var hout = ctx.enqueue_create_host_buffer[DType.float32](R * D)
    var hm = ctx.enqueue_create_host_buffer[DType.float32](R)
    var hz = ctx.enqueue_create_host_buffer[DType.float32](R)
    ctx.enqueue_copy(dst_buf=hout, src_buf=dout)
    ctx.enqueue_copy(dst_buf=hm, src_buf=dm)
    ctx.enqueue_copy(dst_buf=hz, src_buf=dz)
    ctx.synchronize()
    var out_bits: List[UInt32] = [
        1039565469,3193309943,1044067117, 3194409318,3181469939,3191669237,
        3206545408,1061158912,3219128320, 1071644672,3208642560,1059061760,
    ]
    var m_bits: List[UInt32] = [1080033280,1080033280,1077936128,1117782016]
    var z_bits: List[UInt32] = [1092778760,1083708103,1065353216,1065353216]
    var bad = 0
    for i in range(R * D):
        if bitcast[DType.uint32](hout[i]) != out_bits[i]:
            print("output", i, "got", bitcast[DType.uint32](hout[i]), "want", out_bits[i])
            bad += 1
    for i in range(R):
        if bitcast[DType.uint32](hm[i]) != m_bits[i] or bitcast[DType.uint32](hz[i]) != z_bits[i]:
            print("normalizer", i, "m", bitcast[DType.uint32](hm[i]), "want", m_bits[i], "z", bitcast[DType.uint32](hz[i]), "want", z_bits[i])
            bad += 1
    if bad != 0:
        raise Error("attention-v2 differs in " + String(bad) + " cells")
    print("attention_v2_forward_check PASS: TILE=32 tails masks windows extremes digest 05fd2931a1d9934e3913afd3ee4860d23823a8cbfe4c870b3b94ce08a0fac1c1")
    _ = dq^; _ = dk^; _ = dv^; _ = dlo^; _ = dhi^; _ = dout^; _ = dm^; _ = dz^
    _ = hq^; _ = hs^; _ = hv^; _ = hlo^; _ = hhi^; _ = hout^; _ = hm^; _ = hz^
