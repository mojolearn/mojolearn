# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The POLYNOMIAL and TANH epilogues: `polynomial_kernel_nopad`,
`tanh_kernel_nopad`, `PolynomialKernel::evaluate`, `TanhKernel::evaluate`.

PORT OF `cuvs/cpp/src/distance/detail/kernels/kernel_matrices.cu` at cuVS
`6ba2ce2` (`upstream/cuvs-v26.08.00`), lines 18-92 and 136-190. Dense,
row-major, FP32.

**THIS FILE COMPLETES A MIRROR THAT ALREADY EXISTS AND IS PARTIAL.**
`svm/derived/distance/kernel_matrices.mojo` mirrors the SAME upstream file and
ports only its LINEAR and RBF halves; its header says so in one line
("POLYNOMIAL and TANH are NOT ported (refused by name in
`svm_parameter.mojo`; `svm/NOT_IMPLEMENTED.tsv`)"). That refusal is the SVM lane's
and it is correct for the SVM lane, whose solver never reaches those kernels.
Kernel ridge and Nystroem do. **`svm/` is NOT edited by this lane** -- the two
epilogues are ported HERE, beside the callers that need them, and
`kernel_methods/DERIVATION_MAP.tsv` records the file as a SECOND PARTIAL MIRROR
of one upstream file rather than pretending either half is complete.
DEVIATION 1664.

The linear dot product these two epilogues sit on top of is NOT re-spelled
here. `svm/derived/distance/kernel_matrices.mojo::kernel_op` already issues it
through `identical_gemm_into` at `OP_NT`, and `kernel_methods/original/
kernel_matrix.mojo` calls that and then launches one of these. One matrix
product in this repository, under one profile.

THE ROUNDING SEQUENCE. Theirs, for `math_t = float`:

    poly: out = pow(gain * out + offset, exponent)      exponent is `exp_t`,
                                                        an INT at every cuVS
                                                        call site
    tanh: out = tanh(gain * out + offset)

Ours, both modes the same association, the pins under IDENTICAL:

    t    = ftz( identical_mul_add(gain, dot, offset) )
    poly = the ascending repeated product of `degree` copies of `t`
           (DEVIATION 1663; NOT identical_pow, and the reason is arithmetic)
    tanh = ftz( identical_tanh(t) )

# =========================================================================
# DEVIATION 1663: THE POLYNOMIAL POWER IS REPEATED MULTIPLICATION AT AN
# INTEGER DEGREE, AND A NON-INTEGER DEGREE RAISES BY NAME.
#
# THEIRS is `pow(base, exponent)` with `exponent` an `int`
# (`cuvs::distance::kernels::PolynomialKernel<math_t, exp_t>` is instantiated
# with `exp_t = int` from `KernelParams::degree`, and C's `pow` is exact about
# the sign of a negative base raised to an integral power).
#
# OURS CANNOT SAY THAT. `original/numerics.mojo::identical_pow` is
# `portable_powf`, which is `exp(p * log(x))`, and its own docstring records
# that it "returns NaN" for `x < 0` because its consumer's base is a positive
# `-log(u + 1e-20)`. **A polynomial kernel's base is routinely negative**:
# `gain * <x_i, x_j> + coef0` is negative wherever two rows point away from
# each other by more than `coef0 / gain`, which on centered data is about half
# the matrix. Routing the polynomial kernel through `identical_pow` would fill
# half of every Gram matrix with NaN.
#
# So the power is the ascending repeated product, which is exact for a base
# that is exactly representable and its own pinned rounding sequence
# otherwise, and `km_validate_kernel_params` REFUSES a `degree` that is not a
# non-negative integer under BOTH modes rather than silently taking its floor.
# scikit-learn's `degree` is a float and its default is `3`; cuML's is a
# float and its default is `3`; cuVS's is an `int`. The integers are the
# intersection and they are what this lane accepts.
#
# The order is ASCENDING and SERIAL: `acc = 1; acc = acc * t` `degree` times.
# A repeated-squaring ladder is the same value only when the products are
# exact, so it is a SECOND summation order and it is not used.
# =========================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx

from original.numerics import (
    ftz,
    identical_mul,
    identical_mul_add,
    identical_tanh,
)

#: SCHEDULING. Theirs is `128` (`kernel_matrices.cu:144`, `raft::ceildiv<
#: size_t>((size_t)rows * cols, 128)`); one thread owns one cell here as it
#: does there, so the width moves no bit and the checks vary it.
comptime KM_EPILOGUE_TPB = 256

#: The largest `degree` this lane will unroll into repeated products. Above
#: it the answer would still be defined and the cost would stop being
#: reasonable, and -- much worse -- `gain * dot + coef0` raised to a large
#: power overflows float32 long before the arithmetic gets interesting.
#: cuVS's own tests use 2 and 3; scikit-learn's default is 3.
comptime KM_MAX_DEGREE = 32


def polynomial_epilogue_kernel(
    inout_k: MutPointer[Float32, MutAnyOrigin],
    len_in: Int32,
    degree_in: Int32,
    gain: Float32,
    offset: Float32,
):
    """PORT OF `polynomial_kernel_nopad` (`kernel_matrices.cu:27-34`).

    `inout[tid] = pow(gain * inout[tid] + offset, exponent)`, one thread per
    cell. Theirs strides a grid-stride loop over `len`; ours is one thread per
    cell with a bounds test, which is the shape every other elementwise kernel
    in this tree has and which reads no launch geometry into a value.

    The power is DEVIATION 1663's ascending repeated product. `degree == 0`
    returns exactly `1.0`, which is what `pow(x, 0)` returns for every finite
    `x` including a negative one.
    """
    var n = Int(len_in)
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid >= n:
        return
    var dot = ftz(inout_k.unsafe_load(tid))
    var t = ftz(identical_mul_add(gain, dot, offset))
    var acc = Float32(1.0)
    for _ in range(Int(degree_in)):
        acc = ftz(identical_mul(acc, t))
    inout_k.unsafe_store(tid, acc)


def tanh_epilogue_kernel(
    inout_k: MutPointer[Float32, MutAnyOrigin],
    len_in: Int32,
    gain: Float32,
    offset: Float32,
):
    """PORT OF `tanh_kernel_nopad` (`kernel_matrices.cu:66-72`).

    `inout[tid] = tanh(gain * inout[tid] + offset)`, one thread per cell.

    `identical_tanh` is `original/numerics.mojo::portable_tanhf` under
    IDENTICAL and the stdlib's device `tanh` under FAST (IDENTITY_PATHS row
    12). A device `tanh` is a vendor choice in its last bit, and this kernel
    calls it once per cell of an `n x n` matrix, so it is the single largest
    transcendental surface in the lane.
    """
    var n = Int(len_in)
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tid >= n:
        return
    var dot = ftz(inout_k.unsafe_load(tid))
    var t = ftz(identical_mul_add(gain, dot, offset))
    inout_k.unsafe_store(tid, ftz(identical_tanh(t)))
