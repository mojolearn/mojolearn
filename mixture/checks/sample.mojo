# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`GaussianMixture.sample(n_samples)` (2026-09-15): the draws, the device
kernel and the host loop that restates it.

Reference (semantics): scikit-learn `BaseMixture.sample`
(`sklearn/mixture/_base.py`): the per-component counts are one multinomial
draw of `n_samples` over `weights_`, then for component `j` in ascending
order `counts[j]` rows of `multivariate_normal(means_[j], covariances_[j])`
are stacked, and `y` is `j` repeated `counts[j]` times. So `X` comes out
GROUPED BY COMPONENT, ascending, and so does `y`. cuML has no Gaussian
mixture, so there is no GPU reference.

DEVIATION 2791: THE STREAM. scikit-learn draws from one `RandomState`
STREAM, an order three vendors cannot agree on for free. Every draw here is
a pure function of its position, the pattern of DEVIATION 1733 (the
`init_params='random'` responsibilities):

  key      `random_state` as UInt64, low word then high word, the key
           DEVIATION 1733 uses.
  counts   draw `i` in `[0, n_samples)` is `philox4x32_10(ctr, key)` word 0
           with `ctr = (i low, i high, 0, GMM_SAMPLE_COMPONENT_TAG)`, mapped
           to `Float32(w >> 8) * 2^-24` (exact). Its component is the first
           `j < k - 1` with `u < cum[j]`, `cum` the ascending `ftz` fold of
           `weights_`, else `k - 1`; the counts are the tally, which is a
           multinomial draw of `n_samples` over the weights, as theirs is.
  normals  output row `r`, feature `p` reads Box-Muller pair `q = p // 2`
           (DEVIATION 1677's pairing): `philox4x32_10(ctr, key)` with
           `ctr = (r low, r high, q, GMM_SAMPLE_NORMAL_TAG)`, word 0 then
           word 1, each mapped as above, word 0 through `km_guard_unit`
           (DEVIATION 1676), then `km_boxmuller_pair(u1, u2, 1, 0)`
           (RAFT's `box_muller_transform` with the pinned seams); `p` even
           takes the first value, `p` odd the second.

The tags keep the two streams apart from each other and from DEVIATION
1733's (whose `ctr[3]` is 0). The draws are the same bits on every vendor
and at every launch, and they are not scikit-learn's bits: `sample` is
MEANING-COMPATIBLE (the same distribution, the same grouping) and
BIT-DIFFERENT from theirs.

DEVIATION 2792: THE FACTOR. `multivariate_normal` factors the covariance
with an SVD. Here the fitted `precisions_cholesky_` is used as it stands:
it is `P = (L^-1)^T` with `L` the lower Cholesky factor of the covariance,
so `L z` is the `y` that solves the LOWER triangular `P^T y = z`, by forward
substitution in ascending row and column order: `acc = z[i]`, then
`acc = identical_mul_add(-P[j, i], y[j], acc)` for `j < i`, then
`y[i] = identical_div(acc, P[i, i])`, every partial through `ftz`, and the
row is `means_[j] + y`. No second factorization runs, so nothing here can
refuse a model the fit accepted. `X` is float32, the model's dtype, where
scikit-learn's is float64; `y` is int32, `predict`'s dtype.
"""

from max.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast

from checks.numerics import ftz, identical_div, identical_mul_add
from core.philox import philox4x32_10
from kernel_methods.impl.random.rng_device import km_boxmuller_pair, km_guard_unit


#: `ctr[3]` of the component draws, ASCII "COMP". DEVIATION 2791.
comptime GMM_SAMPLE_COMPONENT_TAG: UInt32 = 0x434F4D50
#: `ctr[3]` of the normal draws, ASCII "SAMP". DEVIATION 2791.
comptime GMM_SAMPLE_NORMAL_TAG: UInt32 = 0x53414D50
#: `2^-24` as float32 bits, DEVIATION 1733's scale.
comptime GMM_SAMPLE_TWO_POW_M24_BITS: UInt32 = 0x33800000
#: Threads per block of the device kernel. SCHEDULING ONLY: each thread
#: writes its own row from its own position, with no reduction.
comptime GMM_SAMPLE_TPB = 128


@always_inline
def gmm_sample_key(seed: UInt64) -> SIMD[DType.uint32, 2]:
    return SIMD[DType.uint32, 2](
        UInt32(seed & 0xFFFFFFFF), UInt32((seed >> 32) & 0xFFFFFFFF)
    )


@always_inline
def gmm_sample_unit(w: UInt32) -> Float32:
    """`Float32(w >> 8) * 2^-24`: exact, a 24-bit integer times a power of
    two."""
    return Float32(Int(w >> UInt32(8))) * bitcast[DType.float32](
        GMM_SAMPLE_TWO_POW_M24_BITS
    )


@always_inline
def gmm_sample_normal(key: SIMD[DType.uint32, 2], row: Int, p: Int) -> Float32:
    """The standard normal of output row `row`, feature `p` (DEVIATION
    2791)."""
    var q = p // 2
    var ctr = SIMD[DType.uint32, 4](
        UInt32(row & 0xFFFFFFFF),
        UInt32((row >> 32) & 0xFFFFFFFF),
        UInt32(q & 0xFFFFFFFF),
        GMM_SAMPLE_NORMAL_TAG,
    )
    var draw = philox4x32_10(ctr, key)
    var u1 = km_guard_unit(gmm_sample_unit(draw[0]))
    var u2 = gmm_sample_unit(draw[1])
    var pair = km_boxmuller_pair(u1, u2, Float32(1.0), Float32(0.0))
    if p % 2 == 0:
        return pair[0]
    return pair[1]


def gmm_sample_components(
    weights: List[Float32], k: Int, n_samples: Int, seed: UInt64
) raises -> List[Int32]:
    """The component of every OUTPUT row, grouped ascending: the counts of
    DEVIATION 2791's categorical draws, laid out as scikit-learn stacks
    them. Integer bookkeeping and float32 comparisons only, so the host
    and the device estimator call this one function."""
    if k < 1:
        raise Error("GaussianMixture.sample: the model has no components")
    if n_samples < 1:
        raise Error(
            "GaussianMixture.sample: Invalid value for 'n_samples': "
            + String(n_samples)
            + " . The sampling requires at least one sample."
        )
    var key = gmm_sample_key(seed)
    var cum = List[Float32](length=k, fill=Float32(0.0))
    var acc = Float32(0.0)
    for j in range(k):
        acc = ftz(acc + ftz(weights[j]))
        cum[j] = acc
    var counts = List[Int](length=k, fill=0)
    for i in range(n_samples):
        var ctr = SIMD[DType.uint32, 4](
            UInt32(i & 0xFFFFFFFF),
            UInt32((i >> 32) & 0xFFFFFFFF),
            UInt32(0),
            GMM_SAMPLE_COMPONENT_TAG,
        )
        var u = gmm_sample_unit(philox4x32_10(ctr, key)[0])
        var comp = k - 1
        for j in range(k - 1):
            if u < cum[j]:
                comp = j
                break
        counts[comp] += 1
    var rows = List[Int32](capacity=n_samples)
    for j in range(k):
        for _ in range(counts[j]):
            rows.append(Int32(j))
    return rows^


def gmm_sample_kernel(
    x_out: MutPointer[Float32, MutAnyOrigin],
    comp: MutPointer[Int32, MutAnyOrigin],
    means: MutPointer[Float32, MutAnyOrigin],
    prec: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    d_in: Int32,
    seed_lo: Int32,
    seed_hi: Int32,
):
    """One thread per output row: DEVIATION 2791's normals, DEVIATION 2792's
    forward substitution into `x_out[row]`, then the mean added."""
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(n_in):
        return
    var d = Int(d_in)
    var dd = d * d
    var kc = Int(comp.unsafe_load(row))
    var key = SIMD[DType.uint32, 2](
        seed_lo.cast[DType.uint32](), seed_hi.cast[DType.uint32]()
    )
    for i in range(d):
        var acc = ftz(gmm_sample_normal(key, row, i))
        for j in range(i):
            acc = ftz(
                identical_mul_add(
                    -ftz(prec.unsafe_load(kc * dd + j * d + i)),
                    ftz(x_out.unsafe_load(row * d + j)),
                    acc,
                )
            )
        x_out.unsafe_store(
            row * d + i,
            ftz(identical_div(acc, ftz(prec.unsafe_load(kc * dd + i * d + i)))),
        )
    for i in range(d):
        x_out.unsafe_store(
            row * d + i,
            ftz(
                ftz(means.unsafe_load(kc * d + i))
                + ftz(x_out.unsafe_load(row * d + i))
            ),
        )


def gmm_sample_host(
    weights: List[Float32],
    means: List[Float32],
    prec: List[Float32],
    k: Int,
    d: Int,
    n_samples: Int,
    seed: UInt64,
    mut labels: List[Int32],
) raises -> List[Float32]:
    """`gmm_sample_kernel` on the host, row by row and statement for
    statement, for the CPU binding. `labels` is filled with the component of
    every row."""
    var rows = gmm_sample_components(weights, k, n_samples, seed)
    var key = gmm_sample_key(seed)
    var dd = d * d
    var x = List[Float32](length=n_samples * d, fill=Float32(0.0))
    for row in range(n_samples):
        var kc = Int(rows[row])
        labels[row] = rows[row]
        for i in range(d):
            var acc = ftz(gmm_sample_normal(key, row, i))
            for j in range(i):
                acc = ftz(
                    identical_mul_add(
                        -ftz(prec[kc * dd + j * d + i]), ftz(x[row * d + j]), acc
                    )
                )
            x[row * d + i] = ftz(identical_div(acc, ftz(prec[kc * dd + i * d + i])))
        for i in range(d):
            x[row * d + i] = ftz(ftz(means[kc * d + i]) + ftz(x[row * d + i]))
    return x^
