"""The position-mapped draws: Nystroem's basis rows, RBFSampler's `W` and `b`.

NO UPSTREAM FOR THE ESTIMATORS. cuML ships no `Nystroem` and no `RBFSampler`
at `265b9da` -- `grep -rn 'Nystroem\\|RBFSampler'` over the whole checkout
returns nothing -- so scikit-learn's `sklearn/kernel_approximation.py` is the
SEMANTICS reference and the ORACLE, and nothing in this file is a
transliteration of a device kernel that exists somewhere.

WHAT **IS** PORTED, AND IS PORTED WITHOUT A CHANGE OF ANY KIND, IS THE
GENERATOR AND ITS TRANSFORM:

  - `core/philox.mojo` holds RAFT's `PhiloxGenerator` (cuRAND's
    `curandStatePhilox4_32_10_t`), its `next_float` and its Lemire range
    reduction, transcribed line by line and held to an oracle built by
    COMPILING their generator.
  - `km_boxmuller_pair` below is `raft::random::detail::box_muller_transform`
    (`raft/random/detail/rng_device.cuh:133-142`), five lines, in their order.
  - `resample/mojo_only/index_map.mojo` holds the POSITION MAP -- the derived
    key, the subsequence packing, the single position-mapped uniform and the
    permutation key -- and this file CALLS it rather than re-spelling it.

# =========================================================================
# DEVIATION 1679 -- THIS LANE IMPORTS `resample/mojo_only/index_map.mojo`
# AND SHARES ITS KIND-BYTE NAMESPACE. THE COUPLING IS NAMED, NOT HIDDEN.
#
# WHY THE IMPORT. `index_map.mojo`'s DEVIATION 1690 is exactly the property
# this lane's brief demands ("the row sample must be position-mapped through
# Philox, not drawn from a stream, for the same reason the resampling lane
# needs it"), and it is already written, already argued at length, and
# already gated by `check_index_map_is_positional`, `check_batch_invariance`
# and `check_prefix_stability`. Re-spelling `position_subsequence`,
# `resample_key`, `key_lo`/`key_hi`/`key_join` and `draw_unit_float` here
# would be five copies of a construction whose whole value is that there is
# one of it. `ANTI-DUPLICATION IS THE MAIN RISK IN YOUR LANE` is the brief's
# sentence and this is where it bites hardest.
#
# WHAT THE COUPLING COSTS. `resample_key(seed, kind)` is FNV-1a64 over the
# seed and a KIND BYTE, and the kind byte namespace is now shared between two
# lanes that do not otherwise know about each other. `resample/` spends 1, 2
# and 3 (bootstrap, permutation, Monte Carlo). This lane spends 17, 18 and 19
# and leaves the gap deliberately, so a third lane taking 4 or 5 collides
# with nobody. **A COLLISION WOULD NOT BE AN ERROR, IT WOULD BE A
# CORRELATION**: two uses of one seed sharing a counter stream, which is the
# precise failure `index_map.mojo`'s DEVIATION 1691 exists to prevent and
# which no test distinguishes from a valid draw.
#
# `check_km_kind_bytes_are_disjoint` asserts the three values here differ
# from resample's three, so the namespace claim is checked rather than
# commented.
#
# THE FUNCTION NAMES READ ODDLY HERE AND THAT IS DELIBERATE. `resample_key`
# in a kernel-methods file looks wrong until you know it is a shared
# primitive, and a local alias would hide the dependency the paragraph above
# exists to advertise.
# =========================================================================

# =========================================================================
# DEVIATION 1675 -- THE NORMAL DRAW IS RAFT'S BOX-MULLER, NOT NUMPY'S
# ZIGGURAT, AND NO BIT COMPARISON WITH scikit-learn IS POSSIBLE.
#
# scikit-learn's `RBFSampler.fit` is `random_state.normal(size=(n_features,
# n_components))` (`kernel_approximation.py:381-383`), and
# `numpy.random.RandomState.normal` is the polar (Marsaglia) method over a
# Mersenne Twister while `numpy.random.Generator.normal` is a 256-level
# ziggurat over PCG64. Neither is reproducible from a counter-based generator
# without porting numpy's stream, which is not a thing this repository does
# for any estimator.
#
# So `RBFSampler` is the ONE estimator in this lane whose numbers cannot be
# compared cell by cell with scikit-learn's, and the checks say so rather
# than inventing a comparison. What IS compared:
#
#   - the FEATURE MAP against a float64 HOST replay of OUR draws, per cell
#     (`check_rbf_sampler_vs_oracle`);
#   - the KERNEL APPROXIMATION `z(x).z(y) ~ exp(-gamma |x-y|^2)` against the
#     exact kernel, REPORTED and never asserted, because it is a Monte Carlo
#     estimate whose error is O(1/sqrt(D)) and an assertion on it would be an
#     assertion about a random variable
#     (`check_rbf_sampler_approximates_kernel`, which prints REPORT);
#   - PREFIX STABILITY, bit for bit, which is a property of the MAP and is
#     the strongest thing this arm can assert
#     (`check_random_features_prefix_stability`).
# =========================================================================

# =========================================================================
# DEVIATION 1676 -- RAFT'S BOX-MULLER DOES NOT GUARD `log(0)`. WE DO.
#
# The guard and its full argument live beside the transform, in
# `kernel_methods/ported/random/rng_device.mojo::km_guard_unit`. Summarised
# here because this is the file that CALLS it: `next_float`'s range is
# `[0, 1)` closed at zero, RAFT's `R = sqrt(-2 log(val1))` is `+inf` there,
# and the probability is `2^-24` per pair -- one fit in 256 at a
# `128 x 1024` weight matrix. `+0.0` is replaced by `2^-24`, the smallest
# value the generator itself can return.
#
# CHECKED, not assumed: `check_boxmuller_guard` calls `km_guard_unit` and
# `km_boxmuller_pair` directly at `u1 = +0.0` and requires the guarded pair
# finite and the unguarded pair non-finite. The `KMSAB_NO_BOXMULLER_GUARD`
# arm drops it, and it is classified REPORT in the fixture sweep rather than
# MUST FAIL, because a sweep over ordinary fixtures will not hit a `2^-24`
# event and an arm that "must fail" on a probability-`2^-24` input is an arm
# that fails the gate for the wrong reason.
# =========================================================================

# =========================================================================
# DEVIATION 1677 -- ONE BOX-MULLER PAIR SERVES COMPONENTS `2p` AND `2p+1`.
#
# The transform produces TWO independent standard normals from two uniforms
# and there is no cheaper way to get one. Spending only the first would
# double the RNG arithmetic in the one arm of this lane whose
# arithmetic-intensity story is the RNG itself.
#
# So feature `f`, component `j` reads pair index `p = j // 2` at position
# `(f, p)` and takes `val1` when `j` is even and `val2` when `j` is odd.
# **PREFIX STABILITY SURVIVES**: components `[0, D)` read pairs
# `[0, ceil(D/2))`, and the pair at `(f, p)` is a pure function of
# `(key, f, p)`, so growing `D` adds pairs and moves none. An odd `D`
# discards `val2` of the last pair, which costs nothing and changes no
# earlier component. `check_random_features_prefix_stability` drives
# `D = 64` against `D = 256` and also `D = 65` against `D = 64`, because the
# odd case is where a pairing scheme goes wrong.
# =========================================================================

# =========================================================================
# DEVIATION 1678 -- `sqrt(2 gamma)`, `2 pi` AND `sqrt(2 / D)` ARE HOST
# CONSTANTS, COMPUTED ONCE, PASSED AS KERNEL ARGUMENTS.
#
# `index_map.mojo::draw_uniform_in`'s docstring states the rule this follows
# ("`span` and `lo` are computed ONCE ON THE HOST and passed in, so no kernel
# recomputes a subtraction that two vendors could round apart at a
# subnormal"), and the three constants here are the same shape of hazard:
# `2 / D` for a non-power-of-two `D` is inexact, `sqrt` of it is inexact
# again, and a kernel that recomputes them per thread is a kernel whose
# answer depends on a division being the same division everywhere.
#
# `km_feature_scale` and `km_weight_sigma` are the two host functions, and
# `KMSAB_RF_SCALE_IN_KERNEL` is the arm that moves the computation inside.
# That arm is classified REPORT and is EXPECTED INERT on a column whose
# device division and device sqrt are the same as its host ones, which is
# every column under IDENTICAL by construction. **An expected-inert arm is
# still worth driving**: what it proves is that the constants are not
# secretly a fold or a launch-shaped quantity, and if it ever MOVES on some
# column that is a finding about that column's division.
# =========================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.philox import PhiloxState
from kernel_methods.mojo_only.km_sabotage import (
    KMSAB_NONE,
    km_sabotage_is_kernel_arm,
    sabotage_feature_map_epilogue_kernel,
    sabotage_random_weights_kernel,
)
from kernel_methods.ported.random.rng_device import (
    km_boxmuller_pair,
    km_guard_unit,
    km_two_pi,
    km_unit_float_from,
)
from mojo_only.numerics import (
    ftz,
    identical_cos,
    identical_div,
    identical_mul,
    identical_sqrt,
)
from resample.mojo_only.index_map import (
    draw_permutation_key,
    draw_uniform_in,
    key_hi,
    key_join,
    key_lo,
    permutation_key_lt,
    position_subsequence,
    resample_key,
)


# ===========================================================================
# The kind bytes (DEVIATION 1679). Disjoint from `resample/`'s 1, 2, 3.
# ===========================================================================

#: Nystroem's basis row sample.
comptime KM_KIND_BASIS: UInt64 = 17

#: RBFSampler's random weights `W`.
comptime KM_KIND_RF_WEIGHT: UInt64 = 18

#: RBFSampler's random offsets `b`.
comptime KM_KIND_RF_OFFSET: UInt64 = 19


# ===========================================================================
# Pinned constants
#
# `KM_TWO_PI_BITS`, `KM_MIN_UNIT_BITS`, `km_two_pi()` and `km_min_unit()` are
# NOT here: they moved to `kernel_methods/ported/random/rng_device.mojo`
# beside the RAFT transform that owns them, which is also what breaks the
# import cycle between this file and `km_sabotage.mojo`. Imported above.
# ===========================================================================

#: The O(N^2) rank pass's bound, the same shape of refusal as
#: `resample/mojo_only/index_map.mojo::PERM_MAX_POOLED` and for the same
#: reason: the basis sample ranks every training row against every other by
#: counting a total order, which is `n_samples^2` comparisons on the host.
#: DEVIATION 1672.
comptime KM_MAX_BASIS_POOL = 4096


def km_key(seed: UInt64, kind: UInt64) -> UInt64:
    """`resample_key`, called. DEVIATION 1679 argues the import; this
    one-line forwarder exists so every call site in this lane says
    `km_key(seed, KM_KIND_...)` and a reader sees one derivation."""
    return resample_key(seed, kind)


# ===========================================================================
# RAFT's Box-Muller, spent at positions of this lane's choosing.
#
# The transform itself, its `next_float` and DEVIATION 1676's guard are in
# `kernel_methods/ported/random/rng_device.mojo`. This file decides WHERE in
# the counter space each weight is drawn from and nothing else.
# ===========================================================================


def km_normal_pair(
    key: UInt64, f: Int, p: Int, sigma: Float32
) -> Tuple[Float32, Float32]:
    """The two weights at feature `f`, pair index `p`. DEVIATIONS 1675-1677.

    ONE generator, initialised at position `(f, p)` through
    `resample/mojo_only/index_map.mojo::position_subsequence`, then TWO
    `next_float` words, then RAFT's transform. Pure in `(key, f, p)` and in
    nothing else -- not in `n_components`, not in `n_features`, not in the
    launch, not in what any other thread did.
    """
    var gen = PhiloxState.init(
        key, position_subsequence(UInt64(f), UInt64(p)), UInt64(0)
    )
    var u1 = km_guard_unit(km_unit_float_from(gen))
    var u2 = km_unit_float_from(gen)
    return km_boxmuller_pair(u1, u2, sigma, Float32(0.0))


def km_random_weight(key: UInt64, f: Int, j: Int, sigma: Float32) -> Float32:
    """`random_weights_[f, j]` = `sqrt(2 gamma) * z`, scikit-learn's
    `kernel_approximation.py:381-383`, with `sigma = sqrt(2 gamma)` folded
    into RAFT's transform rather than applied afterwards.

    Component `j` takes `val1` of pair `j // 2` when `j` is even and `val2`
    when it is odd. DEVIATION 1677.
    """
    var pair = km_normal_pair(key, f, j // 2, sigma)
    if j % 2 == 0:
        return pair[0]
    return pair[1]


def km_random_offset(key: UInt64, j: Int) -> Float32:
    """`random_offset_[j]` = `uniform(0, 2 pi)`,
    `kernel_approximation.py:385`.

    `index_map.mojo::draw_uniform_in` CALLED, at replicate 0 and position
    `j`: RAFT's affine map `u * span + lo` with `lo = +0.0` and
    `span = 2 pi`, pinned through `identical_mul_add`. Note their order,
    which is multiply by the span and THEN add the start; distributing it
    differently is a different float.
    """
    return draw_uniform_in(key, 0, j, Float32(0.0), km_two_pi())


# ===========================================================================
# The host constants (DEVIATION 1678)
# ===========================================================================


def km_weight_sigma(gamma: Float32) -> Float32:
    """`(2.0 * gamma) ** 0.5` (`kernel_approximation.py:381`), once, on the
    host, through `identical_sqrt`."""
    return ftz(identical_sqrt(ftz(identical_mul(Float32(2.0), gamma))))


def km_feature_scale(n_components: Int) -> Float32:
    """`(2.0 / n_components) ** 0.5` (`kernel_approximation.py:414`), once,
    on the host, through `identical_div` and `identical_sqrt`."""
    return ftz(
        identical_sqrt(
            ftz(identical_div(Float32(2.0), Float32(n_components)))
        )
    )


# ===========================================================================
# Nystroem's basis row sample (DEVIATIONS 1671, 1672)
# ===========================================================================


def km_basis_indices(
    seed: UInt64, n_samples: Int, n_components: Int
) raises -> List[Int32]:
    """`inds = rnd.permutation(n_samples); basis_inds = inds[:n_components]`
    (`kernel_approximation.py:1050-1051`), position-mapped.

    THEIRS is a Fisher-Yates shuffle over a SEQUENTIAL stream: entry `i` of
    the permutation depends on every swap before it, and the number of stream
    words consumed is itself a function of `n_samples`. `index_map.mojo`'s
    DEVIATION 1690 explains at length why that cannot be made parallel,
    machine-independent and batch-independent at once.

    OURS gives row `j` a 64-bit ordering key at position `(0, j)` --
    `draw_permutation_key`, CALLED, not re-spelled -- and takes the rows whose
    RANK under the total order `(key, index)` is below `n_components`. That
    is a uniformly random permutation prefix with three properties their
    shuffle does not have:

      (a) **PREFIX STABILITY IN `n_components`.** The basis for `q = 64` is
          the first 64 entries of the basis for `q = 256`, in the same order,
          bit for bit. `check_nystroem_basis_prefix_stability` gates it.
      (b) **The key is a pure function of `(seed, j)`**, so the draw is
          embarrassingly parallel with no cross-thread RNG state.
      (c) **Distinctness is free.** The rank is a bijection because the order
          includes the index, so no two rows can claim one rank and the
          sample is without replacement by construction rather than by a
          rejection loop.

    THE RANK PASS IS O(n_samples^2) AND RUNS ON THE HOST. That is the same
    shape and the same bound as `index_map.mojo::permutation_ranks_host`, and
    it is a HOST pass for the reason `decomposition/ported/linalg/detail/
    pca.mojo::eig_and_truncate` gives about its own ordering: sklearn's runs
    on the host too, and the cost is O(n^2) in the SAMPLE COUNT, never in the
    feature count. Above `KM_MAX_BASIS_POOL` it RAISES with the closure
    (DEVIATION 1672).

    Returns the basis row ids IN RANK ORDER, `n_components` of them.
    """
    if n_samples <= 0:
        raise Error(
            "km_basis_indices: n_samples must be positive, got "
            + String(n_samples)
        )
    if n_components <= 0:
        raise Error(
            "km_basis_indices: n_components must be positive, got "
            + String(n_components)
            + ". DEVIATION 1686"
        )
    if n_components > n_samples:
        raise Error(
            "km_basis_indices: n_components ("
            + String(n_components)
            + ") exceeds n_samples ("
            + String(n_samples)
            + "). scikit-learn WARNS and clamps to n_samples"
            " (kernel_approximation.py:1038-1047, 'XXX should we just"
            " bail?'); this lane RAISES, because a fit that silently"
            " returns a different number of components than it was asked"
            " for produces an embedding whose width is not the width the"
            " caller allocated for. DEVIATION 1673; pass"
            " n_components = n_samples explicitly for the full kernel"
        )
    if n_samples > KM_MAX_BASIS_POOL:
        raise Error(
            "km_basis_indices: n_samples "
            + String(n_samples)
            + " exceeds KM_MAX_BASIS_POOL = "
            + String(KM_MAX_BASIS_POOL)
            + ". The rank pass counts a total order over every pair of rows,"
            " which is n_samples^2 host comparisons. To close this refusal,"
            " replace the counting rank with a pinned segmented sort over"
            " the 64-bit composite key -- the construction"
            " neighbors/mojo_only/select_radix_identical.mojo uses for its"
            " (distance, index) key -- and re-gate"
            " check_nystroem_basis_prefix_stability at the larger size."
            " DEVIATION 1672"
        )

    var key = km_key(seed, KM_KIND_BASIS)
    var keys = List[UInt64]()
    for j in range(n_samples):
        keys.append(draw_permutation_key(key, 0, j))

    var out = List[Int32]()
    for _ in range(n_components):
        out.append(Int32(0))
    for j in range(n_samples):
        var rank = 0
        for l in range(n_samples):
            if permutation_key_lt(keys[l], l, keys[j], j):
                rank += 1
        if rank < n_components:
            out[rank] = Int32(j)
    return out^


# ===========================================================================
# The kernels. One thread per OUTPUT CELL, nothing shared, nothing folded.
# ===========================================================================

#: SCHEDULING.
comptime KM_RF_TPB = 256


def random_weights_kernel(
    w_out: MutPointer[Float32, MutAnyOrigin],
    lo_bits: Int32,
    hi_bits: Int32,
    n_features_in: Int32,
    n_components_in: Int32,
    sigma: Float32,
):
    """Materialise `W[n_features x n_components]`, row-major.

    Thread `t` owns cell `(f, j) = (t / D, t % D)` and computes it from
    `(key, f, j)` alone. No thread reads another's value, no counter is
    shared, and the answer does not depend on how many threads ran.
    """
    var d = Int(n_components_in)
    var total = Int(n_features_in) * d
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= total:
        return
    var f = t // d
    var j = t - f * d
    w_out.unsafe_store(t, km_random_weight(key_join(lo_bits, hi_bits), f, j, sigma))


def random_offsets_kernel(
    b_out: MutPointer[Float32, MutAnyOrigin],
    lo_bits: Int32,
    hi_bits: Int32,
    n_components_in: Int32,
):
    """Materialise `b[n_components]` = `uniform(0, 2 pi)`."""
    var d = Int(n_components_in)
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j >= d:
        return
    b_out.unsafe_store(j, km_random_offset(key_join(lo_bits, hi_bits), j))


def feature_map_epilogue_kernel(
    proj_io: MutPointer[Float32, MutAnyOrigin],
    b_in: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_components_in: Int32,
    scale: Float32,
):
    """scikit-learn's `transform`, `kernel_approximation.py:411-414`:

        projection = safe_sparse_dot(X, self.random_weights_)
        projection += self.random_offset_
        np.cos(projection, projection)
        projection *= (2.0 / self.n_components) ** 0.5

    ONE THREAD PER CELL, and the four lines in their order. The dot product
    is NOT here: the caller has already run `identical_gemm_into` at `OP_NN`
    into `proj_io`, so this kernel is `add`, `cos`, `mul`.

    THE ORDER IS NOT FREE. `cos(p + b) * scale` and `cos(p) * scale + ...`
    are different functions; `scale * cos(p + b)` and `cos(p + b) * scale`
    are the same bits by commutativity but the ADD and the MULTIPLY cannot be
    exchanged. Their sequence is transcribed and `identical_cos` carries
    IDENTITY_PATHS row 12.
    """
    var d = Int(n_components_in)
    var total = Int(n_rows_in) * d
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= total:
        return
    var j = t % d
    var p = ftz(proj_io.unsafe_load(t))
    var shifted = ftz(p + ftz(b_in.unsafe_load(j)))
    proj_io.unsafe_store(
        t, ftz(identical_mul(identical_cos(shifted), scale))
    )


# ===========================================================================
# Launches
# ===========================================================================


def km_random_weights(
    ctx: DeviceContext,
    mut w: DeviceBuffer[DType.float32],
    seed: UInt64,
    n_features: Int,
    n_components: Int,
    sigma: Float32,
    tpb: Int = KM_RF_TPB,
    sabotage: Int = KMSAB_NONE,
) raises:
    """`W = sqrt(2 gamma) * normal(size=(n_features, n_components))`.
    ASYNCHRONOUS.

    At `KMSAB_NONE` this launches the production kernel and the sabotage file
    is not reached. When `sabotage` names a kernel arm it launches
    `km_sabotage.mojo`'s COPY, so no production kernel here carries a
    sabotage branch (DEVIATION 1687)."""
    if n_features <= 0 or n_components <= 0:
        return
    var key = km_key(seed, KM_KIND_RF_WEIGHT)
    var total = n_features * n_components
    if km_sabotage_is_kernel_arm(sabotage):
        ctx.enqueue_function[sabotage_random_weights_kernel](
            w.unsafe_ptr(),
            key_lo(key),
            key_hi(key),
            Int32(n_features),
            Int32(n_components),
            sigma,
            Int32(sabotage),
            grid_dim=((total + tpb - 1) // tpb, 1, 1),
            block_dim=(tpb, 1, 1),
        )
        return
    ctx.enqueue_function[random_weights_kernel](
        w.unsafe_ptr(),
        key_lo(key),
        key_hi(key),
        Int32(n_features),
        Int32(n_components),
        sigma,
        grid_dim=((total + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )


def km_random_offsets(
    ctx: DeviceContext,
    mut b: DeviceBuffer[DType.float32],
    seed: UInt64,
    n_components: Int,
    tpb: Int = KM_RF_TPB,
) raises:
    """`b = uniform(0, 2 pi, size=n_components)`. ASYNCHRONOUS."""
    if n_components <= 0:
        return
    var key = km_key(seed, KM_KIND_RF_OFFSET)
    ctx.enqueue_function[random_offsets_kernel](
        b.unsafe_ptr(),
        key_lo(key),
        key_hi(key),
        Int32(n_components),
        grid_dim=((n_components + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )


def km_feature_map_epilogue(
    ctx: DeviceContext,
    mut proj: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_components: Int,
    scale: Float32,
    tpb: Int = KM_RF_TPB,
    sabotage: Int = KMSAB_NONE,
) raises:
    """`proj = sqrt(2/D) * cos(proj + b)`, in place. ASYNCHRONOUS."""
    if n_rows <= 0 or n_components <= 0:
        return
    var total = n_rows * n_components
    if km_sabotage_is_kernel_arm(sabotage):
        ctx.enqueue_function[sabotage_feature_map_epilogue_kernel](
            proj.unsafe_ptr(),
            b.unsafe_ptr(),
            Int32(n_rows),
            Int32(n_components),
            scale,
            Int32(sabotage),
            grid_dim=((total + tpb - 1) // tpb, 1, 1),
            block_dim=(tpb, 1, 1),
        )
        return
    ctx.enqueue_function[feature_map_epilogue_kernel](
        proj.unsafe_ptr(),
        b.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_components),
        scale,
        grid_dim=((total + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )


# ===========================================================================
# Host mirrors. ORACLES, not a CPU path (PORTING_RULES 0b-ii): the draws are
# integer arithmetic into a pinned float transform, so a host mirror is bit
# for bit the device's answer BY CONSTRUCTION and a gate that says so is
# worth having.
# ===========================================================================


def km_random_weights_host(
    seed: UInt64, n_features: Int, n_components: Int, sigma: Float32
) -> List[Float32]:
    """`random_weights_kernel` on the host, in index order."""
    var key = km_key(seed, KM_KIND_RF_WEIGHT)
    var out = List[Float32]()
    for f in range(n_features):
        for j in range(n_components):
            out.append(km_random_weight(key, f, j, sigma))
    return out^


def km_random_offsets_host(seed: UInt64, n_components: Int) -> List[Float32]:
    """`random_offsets_kernel` on the host, in index order."""
    var key = km_key(seed, KM_KIND_RF_OFFSET)
    var out = List[Float32]()
    for j in range(n_components):
        out.append(km_random_offset(key, j))
    return out^


# `key_lo`, `key_hi` and `key_join` are `index_map.mojo`'s, imported and not
# re-spelled. THEIR MASKS ARE NOT DECORATION: Mojo has been MEASURED
# sign-extending `Int32 -> UInt32 -> UInt64` through a `var`, and the failure
# hides behind the `<< 32` on the way back because the bad bits shift out
# (`core/philox.mojo`). A local re-spelling of those three lines is exactly
# the duplication this lane exists to not make.
