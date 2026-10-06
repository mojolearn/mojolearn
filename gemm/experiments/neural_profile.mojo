# SPDX-License-Identifier: Apache-2.0
"""NN03/NN04: explicit neural research profiles; never a global GEMM switch.

The pure profile is shared by neural model dispatch and its host mirrors.
Every profile uses this SAME scalar arithmetic body on host and device. A
profile may change bits across versions, never between vendors in one run.
No compilation, identity, quality or timing has been run for this source.
"""
from std.sys.compile import is_defined, get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from checks.rtf_seam import rtf_mul_add
from gemm.contract import CONTRACT_K_LEAF_MIN, CONTRACT_MAX_LEAVES, OP_NN, OP_NT, OP_TN

comptime NEURAL_EXPERIMENTS_ALLOWED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
# Separate defines isolate leaf length from accumulator-chain experiments.
# All OFF pending four-column identity, neural quality and BOTH voting vendors'
# full-workload A/Bs. No inherited I04 component result admits these profiles.
comptime NN03 = NEURAL_EXPERIMENTS_ALLOWED and is_defined["MOJOLEARN_IDN_NEURAL_NN03"]()
comptime NN04 = NEURAL_EXPERIMENTS_ALLOWED and is_defined["MOJOLEARN_IDN_NEURAL_NN04"]()
comptime NEURAL_PROFILE_CHANGED = NN03 or NN04
comptime NEURAL_LEAF = get_defined_int["MOJOLEARN_IDN_NEURAL_LEAF",128]() if NN03 else CONTRACT_K_LEAF_MIN
comptime NEURAL_CHAINS = get_defined_int["MOJOLEARN_IDN_NEURAL_CHAINS",2]() if NN04 else 1


def neural_partition[MIN_LEAF: Int](k: Int) -> Tuple[Int, Int]:
    """The cap is numerical profile data; it never reads m/n or hardware."""
    comptime assert MIN_LEAF == 64 or MIN_LEAF == 128 or MIN_LEAF == 256, "profile leaf"
    if k <= 0:
        return (1, 0)
    var leaf = min(k, MIN_LEAF)
    if (k + leaf - 1) // leaf > CONTRACT_MAX_LEAVES:
        leaf = (k + CONTRACT_MAX_LEAVES - 1) // CONTRACT_MAX_LEAVES
    return (leaf, (k + leaf - 1) // leaf)


def neural_strides(op: Int, m: Int, n: Int, k: Int) -> Tuple[Int, Int, Int, Int]:
    return (1 if op == OP_TN else k, m if op == OP_TN else 1,
            1 if op == OP_NT else n, k if op == OP_NT else 1)


def neural_validate(m: Int, n: Int, k: Int, op: Int) raises:
    if m < 0 or n < 0 or k < 0:
        raise Error("negative neural GEMM dimension")
    if op != OP_NN and op != OP_NT and op != OP_TN:
        raise Error("neural GEMM supports NN, NT and TN")
    # Device kernel index arguments are Int32; do not silently narrow them.
    if m > 2147483647 or n > 2147483647 or k > 2147483647:
        raise Error("neural GEMM dimension exceeds Int32")
    if m*n > 2147483647 or m*k > 2147483647 or n*k > 2147483647:
        raise Error("neural GEMM linear index exceeds Int32")


@always_inline
def neural_fold_push[SLOTS: Int](mut stack: SIMD[DType.float32, SLOTS], mut occupied: Int,
                                  value: Float32):
    var v = value
    var placed = False
    comptime for level in range(SLOTS):
        if not placed:
            if (occupied & (1 << level)) != 0:
                v = ftz(ftz(stack[level]) + ftz(v))
                occupied -= 1 << level
            else:
                stack[level] = v
                occupied += 1 << level
                placed = True


@always_inline
def neural_fold_drain[SLOTS: Int](stack: SIMD[DType.float32, SLOTS], occupied: Int) -> Float32:
    var result = Float32(0)
    var have = False
    comptime for level in range(SLOTS):
        if (occupied & (1 << level)) != 0:
            if have:
                result = ftz(ftz(stack[level]) + ftz(result))
            else:
                result = stack[level]
                have = True
    return ftz(result)


@always_inline
def neural_merge_chains[CHAINS: Int](acc: SIMD[DType.float32,CHAINS]) -> Float32:
    comptime if CHAINS == 1:
        return ftz(acc[0])
    elif CHAINS == 2:
        return ftz(ftz(acc[0])+ftz(acc[1]))
    else:
        return ftz(ftz(ftz(acc[0])+ftz(acc[1]))+ftz(ftz(acc[2])+ftz(acc[3])))


@always_inline
def neural_leaf[CHAINS: Int](
    a: MutPointer[Float32, MutAnyOrigin], b: MutPointer[Float32, MutAnyOrigin],
    row: Int, col: Int, begin: Int, end: Int,
    asi: Int, asp: Int, bsp: Int, bsj: Int, real_k: Int = -1,
) -> Float32:
    """Versioned leaf graph: chain q owns begin+q+j*CHAINS in ascending j.

    Every chain starts at +0; unused chains of a ragged leaf still contribute
    their specified +0 in the final adjacent-pair merge. That is explicit
    profile arithmetic, not masked padding of the incumbent one-chain graph.
    """
    comptime assert CHAINS == 1 or CHAINS == 2 or CHAINS == 4, "profile chains"
    var acc = SIMD[DType.float32, CHAINS](0.0)
    for base in range(begin, end, CHAINS):
        comptime for chain in range(CHAINS):
            var p = base + chain
            if p < end:
                var av = Float32(0)
                var bv = Float32(0)
                if real_k<0 or p<real_k:
                    av = ftz(a.unsafe_load(row*asi+p*asp))
                    bv = ftz(b.unsafe_load(p*bsp+col*bsj))
                # Explicit logical +0 suffix still participates in the
                # selected chain; never shorten k or move leaf boundaries.
                acc[chain] = rtf_mul_add(av,bv,acc[chain])
    return neural_merge_chains[CHAINS](acc)


@always_inline
def neural_cell[CHAINS: Int, SLOTS: Int = 16](
    a: MutPointer[Float32, MutAnyOrigin], b: MutPointer[Float32, MutAnyOrigin],
    row: Int, col: Int, k: Int, leaf: Int, leaves: Int,
    asi: Int, asp: Int, bsp: Int, bsj: Int, real_k: Int = -1,
) -> Float32:
    var stack = SIMD[DType.float32, SLOTS](0.0)
    var occupied = 0
    for t in range(leaves):
        var value = neural_leaf[CHAINS](a,b,row,col,t*leaf,min((t+1)*leaf,k),asi,asp,bsp,bsj,real_k)
        neural_fold_push[SLOTS](stack,occupied,value)
    return neural_fold_drain[SLOTS](stack,occupied)


def neural_profile_host[MIN_LEAF: Int = 128, CHAINS: Int = 1](
    c: MutPointer[Float32, MutAnyOrigin], a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin], m: Int, n: Int, k: Int, op: Int,
) raises:
    neural_validate(m,n,k,op)
    var part = neural_partition[MIN_LEAF](k)
    var st = neural_strides(op,m,n,k)
    for row in range(m):
        for col in range(n):
            c.unsafe_store(row*n+col,neural_cell[CHAINS](a,b,row,col,k,part[0],part[1],
                st[0],st[1],st[2],st[3]))


def neural_profile_ab_host[LEAF: Int = 128, CHAINS: Int = 1, CANDIDATE: Bool = False](
    c: MutPointer[Float32, MutAnyOrigin], a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin], m: Int, n: Int, k: Int, op: Int,
) raises:
    comptime L = LEAF if CANDIDATE and NN03 else CONTRACT_K_LEAF_MIN
    comptime C = CHAINS if CANDIDATE and NN04 else 1
    neural_profile_host[L,C](c,a,b,m,n,k,op)


def neural_profile_label[LEAF: Int = 128, CHAINS: Int = 1, CANDIDATE: Bool = False]() -> String:
    comptime L = LEAF if CANDIDATE and NN03 else CONTRACT_K_LEAF_MIN
    comptime C = CHAINS if CANDIDATE and NN04 else 1
    return "neural.fp32.leaf" + String(L) + ".chains" + String(C) + ".pair-tree.v1"
