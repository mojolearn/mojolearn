# SPDX-License-Identifier: Apache-2.0
"""Actual stable radix infrastructure: two/four-pass comparisons,
all-equal and high-bit keys, stable original-position carry, alternating
live lengths and untouched capacity tails. Contract is unsigned UInt32
keys; floating NaN/signed-zero preprocessing remains each caller's policy."""
from max.gpu.host import DeviceContext
from std.sys.compile import is_defined
from core.stable_radix_digits import stable_nibble_pairs_u32
from experiments.performance_ideas.I19.float_check import check_float_ragged
from experiments.performance_ideas.I19.quantile_check import check_quantile_caller
from experiments.performance_ideas.I19.categories_check import check_categories
from core.stable_radix_sort import stable_radix_sort_pairs_u32, stable_radix_counts_len, stable_radix_bsum_len

def check(ctx: DeviceContext,n: Int,bits: Int,all_equal: Bool) raises:
    var keys = List[UInt32]()
    var values = List[UInt32]()
    var mask = UInt32(0xffffffff) if bits==32 else (UInt32(1)<<UInt32(bits))-UInt32(1)
    for i in range(n+17):
        keys.append((UInt32(7) if all_equal else UInt32((i*40503)^((i%29)*65537))) & mask if i<n else UInt32(0xdeadbeef))
        values.append(UInt32(i) if i<n else UInt32(0xcafebabe))
    var baseline_k = List[UInt32]()
    var baseline_v = List[UInt32]()
    for arm in range(2):
        var dk = ctx.enqueue_create_buffer[DType.uint32](n+17)
        var dv = ctx.enqueue_create_buffer[DType.uint32](n+17)
        var tk = ctx.enqueue_create_buffer[DType.uint32](n+17)
        var tv = ctx.enqueue_create_buffer[DType.uint32](n+17)
        var counts = ctx.enqueue_create_buffer[DType.int32](stable_radix_counts_len(n))
        var bsum = ctx.enqueue_create_buffer[DType.int32](stable_radix_bsum_len(n))
        ctx.enqueue_copy(dst_buf=dk,src_ptr=keys.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=dv,src_ptr=values.unsafe_ptr())
        # NEVER RUN — PENDING VALIDATION
        comptime if is_defined["MOJOLEARN_IDN_RAGGED_RADIX_NIBBLE"]():
            if arm==1:
                # NEVER RUN — PENDING VALIDATION
                comptime if is_defined["MOJOLEARN_IDN_RADIX_TILE128"]():
                    stable_nibble_pairs_u32[128](ctx,n,bits,dk,dv,tk,tv,counts,bsum)
                else:
                    stable_nibble_pairs_u32[256](ctx,n,bits,dk,dv,tk,tv,counts,bsum)
            else:
                stable_radix_sort_pairs_u32(ctx,n,32,dk,dv,tk,tv,counts,bsum)
        else:
            stable_radix_sort_pairs_u32(ctx,n,32 if arm==0 else bits,dk,dv,tk,tv,counts,bsum)
        # Reuse the same workspace without a host roundtrip; idempotence
        # must hold, including value carry within every equal-key run.
        # NEVER RUN — PENDING VALIDATION
        comptime if is_defined["MOJOLEARN_IDN_RAGGED_RADIX_NIBBLE"]():
            if arm==1:
                # NEVER RUN — PENDING VALIDATION
                comptime if is_defined["MOJOLEARN_IDN_RADIX_TILE128"]():
                    stable_nibble_pairs_u32[128](ctx,n,bits,dk,dv,tk,tv,counts,bsum)
                else:
                    stable_nibble_pairs_u32[256](ctx,n,bits,dk,dv,tk,tv,counts,bsum)
            else:
                stable_radix_sort_pairs_u32(ctx,n,32,dk,dv,tk,tv,counts,bsum)
        else:
            stable_radix_sort_pairs_u32(ctx,n,32 if arm==0 else bits,dk,dv,tk,tv,counts,bsum)
        var hk = ctx.enqueue_create_host_buffer[DType.uint32](n+17)
        var hv = ctx.enqueue_create_host_buffer[DType.uint32](n+17)
        ctx.enqueue_copy(dst_ptr=hk.unsafe_ptr(),src_buf=dk)
        ctx.enqueue_copy(dst_ptr=hv.unsafe_ptr(),src_buf=dv)
        ctx.synchronize()
        for i in range(n):
            if keys[Int(hv[i])]!=hk[i] or (i>0 and (hk[i]<hk[i-1] or (hk[i]==hk[i-1] and hv[i]<=hv[i-1]))):
                raise Error("I19 key/value stable pairing failed")
            if arm==0:
                baseline_k.append(hk[i]); baseline_v.append(hv[i])
            elif baseline_k[i]!=hk[i] or baseline_v[i]!=hv[i]:
                raise Error("I19 skipped digit changed exact output")
        for i in range(n,n+17):
            if hk[i]!=UInt32(0xdeadbeef) or hv[i]!=UInt32(0xcafebabe):
                raise Error("I19 radix wrote beyond live capacity")
        _ = dk^; _ = dv^; _ = tk^; _ = tv^; _ = counts^; _ = bsum^

# NEVER RUN — PENDING VALIDATION
def main() raises:
    var ctx = DeviceContext()
    check_float_ragged(ctx)
    check_quantile_caller(ctx)
    check_categories(ctx)
    for n in [31,33,513,31]:
        for bits in [8,15,17,32]:
            check(ctx,n,bits,False)
            check(ctx,n,bits,True)
    print("I19 PASS cases=32 pass_arms=2 scratch_reuses=2 stable_carry_and_capacity")
