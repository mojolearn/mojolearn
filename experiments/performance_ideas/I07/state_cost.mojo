# SPDX-License-Identifier: Apache-2.0
"""Explicit experiment budget for canonical retained attention state.

Cost weights are supplied by the experiment, not guessed from a board or
sequence threshold. This model accounts for incremental Float32 exponent
or probability storage and score recomputation work; it does not replace
measured complete-step acceptance or the input/checkpoint lifetime contract.
"""

# I07 experiment qualification pending: compile/fixtures do not establish
# four-column identity or NVIDIA+AMD full-operation speed. Existing promoted
# defaults stay unchanged; this campaign attributes explicit experiment arms.
def _bounded_product(a: UInt64,b: UInt64) raises -> UInt64:
    if b!=0 and a>UInt64(0xffffffffffffffff)//b:
        raise Error("attention state model: cost overflow")
    return a*b

def state_cost(batch: Int,heads: Int,queries: Int,keys: Int,head_dim: Int,
               budget_bytes: UInt64,byte_units: UInt64,fma_units: UInt64,
               exp_units: UInt64) raises -> Tuple[UInt64,UInt64,Bool]:
    if batch<=0 or heads<=0 or queries<=0 or keys<=0 or head_dim<=0:
        raise Error("attention state model: dimensions must be positive")
    if byte_units==0 or fma_units==0 or exp_units==0:
        raise Error("attention state model: cost units must be positive")
    var pairs = _bounded_product(_bounded_product(UInt64(batch),UInt64(heads)),
                                 _bounded_product(UInt64(queries),UInt64(keys)))
    var retained_bytes = _bounded_product(pairs,UInt64(4))
    # Retained state is written in forward and read in backward. A score
    # replay has head_dim FMAs plus one exponential per query/key pair.
    var traffic = _bounded_product(_bounded_product(retained_bytes,UInt64(2)),byte_units)
    var fmas = _bounded_product(UInt64(head_dim),fma_units)
    if fmas>UInt64(0xffffffffffffffff)-exp_units:
        raise Error("attention state model: cost overflow")
    var recompute = _bounded_product(pairs,fmas+exp_units)
    return (retained_bytes,recompute,retained_bytes<=budget_bytes and traffic<=recompute)
