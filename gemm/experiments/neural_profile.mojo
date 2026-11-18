# SPDX-License-Identifier: Apache-2.0
"""NN03/NN04: explicit neural research profiles; never a global GEMM switch.

The pure profile is shared by neural model dispatch and its host mirrors.
Every profile uses this SAME scalar arithmetic body on host and device. A
profile may change bits across versions, never between vendors in one run.
No compilation, identity, quality or timing has been run for this source.
"""

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from checks.rtf_seam import rtf_mul_add
from gemm.contract import (
    CONTRACT_K_LEAF_MIN, CONTRACT_MAX_LEAVES, OP_NN, OP_NT, OP_TN,
    GEMM_LEAF_ARM, GEMM_LEAF_ARM_NEURAL128, GEMM_LEAF_ARM_NEURAL256,
)

# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (its terms here: every IDENTICAL switch off at once; EXP:2438 old path) deleted 2026-11-15 by lane/expdel2-25-neural; code recoverable at 5cde955daf.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-25-neural.patch.
comptime NEURAL_EXPERIMENTS_ALLOWED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
)
# Leaf length and accumulator chains stay separate controls.
# NN03 = the neural arms (1 neural128, 2 neural256) of the ONE leaf switch
# MOJOLEARN_IDN_GEMM_LEAF (gemm/contract.mojo); arm 3 (all256) changes the
# contract for every caller instead and leaves this profile unchanged.
# TOMBSTONE: MOJOLEARN_IDN_NEURAL_CHAINS (NN04, 2|4 accumulator chains: opt-in arm, no A/B recorded; one chain kept) deleted 2026-11-15 by lane/expdel2-25-neural; code recoverable at a9109164e5.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_CHAINS.patch; record in docs/TOMBSTONES.md.
# The former enable defines IDN_NEURAL_NN03/IDN_NEURAL_NN04/IDN_NEURAL_LEAF are gone.
# All OFF pending four-column identity, neural quality and BOTH voting vendors'
# full-workload A/Bs. No inherited I04 component result admits these profiles.
comptime NN03 = NEURAL_EXPERIMENTS_ALLOWED and (
    GEMM_LEAF_ARM == GEMM_LEAF_ARM_NEURAL128 or GEMM_LEAF_ARM == GEMM_LEAF_ARM_NEURAL256
)
comptime NN04 = False
comptime NEURAL_PROFILE_CHANGED = NN03 or NN04
comptime NEURAL_LEAF = (
    256 if NN03 and GEMM_LEAF_ARM == GEMM_LEAF_ARM_NEURAL256
    else (128 if NN03 else CONTRACT_K_LEAF_MIN)
)
comptime NEURAL_CHAINS = 1


def neural_partition[MIN_LEAF: Int](k: Int) -> Tuple[Int, Int]:
    """The cap is numerical profile data; it never reads m/n or hardware."""
    # MIN_LEAF == CONTRACT_K_LEAF_MIN: NEURAL_LEAF is the contract's leaf when no
    # neural arm is on, and callers instantiate this with it even on a branch
    # they do not take; under -D MOJOLEARN_GEMM_FAST_LEAF512 that is 512.
    comptime assert (
        MIN_LEAF == 128 or MIN_LEAF == 256 or MIN_LEAF == CONTRACT_K_LEAF_MIN
    ), "profile leaf (leaf 64 deleted: I04 loser)"
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


