"""The device machinery k-means|| gets from vendor libraries, named call by call.

NOT A PORT of one file. `initScalableKMeansPlusPlus`
(`cuvs/src/cluster/detail/kmeans.cuh:568-785`) leans on four vendor calls that
have no shipped Mojo counterpart, and each kernel here names the one it
replaces:

    raft::random::uniform            -> `scalable_uniform` inside
                                        `sample_flags_kernel` (deviation:
                                        counter-hash, not Philox; see below)
    cub::DeviceSelect::If            -> `sample_flags_kernel` + the existing
                                        three-stage scan from
                                        `mojo_only/plus_plus.mojo` +
                                        `select_scatter_kernel`
    thrust::for_each_n (flag update) -> folded into `select_scatter_kernel`
    cub::DeviceHistogram::HistogramEven (`countLabels`,
                                        `kmeans_common.cuh:95-135`)
                                     -> `count_labels_kernel`

THE SELECTION IS A SCAN, AND THE SCAN IS THE ONE k-MEANS++ ALREADY HAS
----------------------------------------------------------------------
`cub::DeviceSelect::If` is stable compaction: it keeps the passing elements
in index order and reports the count. Flags in {0,1}, the inclusive scan of
the flags, and a scatter at `csum[i] - 1` is the same operation, built out of
`chunk_sums_kernel` / `scan_chunk_offsets_kernel` /
`write_inclusive_scan_kernel`, which `check_device_inclusive_scan` already
holds to a host scan element by element. Nothing new to trust in the scan.

The scan runs in Float32, so its counts are exact only while they fit a
24-bit integer. That bounds `n_samples` at 2^24 and the DRIVER RAISES above
it rather than silently mis-scattering (PORTING.md 48). The same bound covers
`count_labels_kernel`: counts are sums of exact 1.0s, so every intermediate
is an integer below 2^24 and float atomic adds of integers commute EXACTLY,
which is why the count is deterministic without any ordering guarantee.

THE RANDOMNESS IS A COUNTER HASH, AND WHY THAT IS THE HONEST CHOICE HERE
------------------------------------------------------------------------
Their per-sample uniforms come from `raft::random::uniform` -- a Philox-style
COUNTER-BASED generator on device (`raft/random/rng_device.cuh`). No Mojo
counterpart produces that stream (PORTING.md 17 already prices the host-RNG
half of this). What must be preserved is the shape of the mechanism: the
host may not manufacture O(rows) randomness (`HOST_AND_DEVICE.md`), and the
draw must be a pure function of (seed, index) so the same seed gives the
same fit. `scalable_uniform` is splitmix64 -- the same finalizer
`detail/kmeans.mojo::HostRng` already uses, chosen there because it is short
enough to audit -- used as a counter hash: the host draws ONE 64-bit seed
per round, the device hashes (seed, i). Same mechanism class as Philox
(counter-based, stateless per element), different stream. PORTING.md 47.

`scalable_uniform` and `scalable_keep` are plain `def`s callable from host
code too, ON PURPOSE: `check_scalable_sampling_selection` replays the exact
predicate on the host and holds the device selection to it element by
element, which is only meaningful if both sides run the same expression.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx


def scalable_uniform(seed_lo: Int32, seed_hi: Int32, i: Int) -> Float32:
    """One uniform in [0, 1) as a pure function of (round seed, sample index).

    Replaces `raft::random::uniform` at `detail/kmeans.cuh:689-690` for
    k-means|| step 3 (PORTING.md 47). splitmix64 of `seed + i * golden`,
    top 24 bits scaled by 2^-24, so the value is exactly representable and
    identical on host and device.
    """
    var seed = (
        seed_hi.cast[DType.uint32]().cast[DType.uint64]() << 32
    ) | seed_lo.cast[DType.uint32]().cast[DType.uint64]()
    var z = seed + UInt64(i) * 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float32(z >> 40) * Float32(5.9604644775390625e-08)  # 2^-24


def scalable_keep(
    dist: Float32, psi: Float32, lk: Float32, u: Float32
) -> Bool:
    """`SamplingOp::operator()`, `kmeans_common.cuh:73-81`, minus the flag.

        prob_x = (oversampling_factor * n_clusters * a.value) / cluster_cost
        return !flag[a.key] && (prob_x > rnd[a.key])

    The flag half lives in `sample_flags_kernel`, which owns the flag
    pointer. `lk` is `oversampling_factor * n_clusters`, formed ONCE on the
    host in Float64 and passed down as Float32: theirs forms the product in
    double per element and truncates the result to DataT, and Apple silicon
    has no device Float64 to copy that with (PORTING.md 47). The comparison
    is strict `>`, which matters: a zero probability can never be selected,
    whatever the draw.
    """
    var prob = lk * dist / psi
    return prob > u


def sample_flags_kernel(
    flags: MutPointer[Float32, MutAnyOrigin],
    min_dist: MutPointer[Float32, MutAnyOrigin],
    is_centroid: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    psi_in: Float32,
    lk_in: Float32,
    seed_lo: Int32,
    seed_hi: Int32,
):
    """Step 3's independent per-point trial, one thread per sample.

    The predicate half of `cub::DeviceSelect::If`'s `select_op`
    (`sampleCentroids`, `kmeans_common.cuh:228-266`): flags[i] is 1.0 when
    point i is drawn into this round's C'. The scan over these flags is the
    compaction.
    """
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var u = scalable_uniform(seed_lo, seed_hi, i)
    var keep = is_centroid.unsafe_load(i) == Int32(0) and scalable_keep(
        min_dist.unsafe_load(i), psi_in, lk_in, u
    )
    flags.unsafe_store(i, Float32(1.0) if keep else Float32(0.0))


def select_scatter_kernel(
    out_index: MutPointer[UInt32, MutAnyOrigin],
    is_centroid: MutPointer[Int32, MutAnyOrigin],
    flags: MutPointer[Float32, MutAnyOrigin],
    csum: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """The compaction write of `cub::DeviceSelect::If`, plus the flag update.

    `csum` is the INCLUSIVE scan of the flags, so a selected sample's rank is
    `csum[i] - 1` and the write is collision-free by construction. The flag
    update is theirs too: `sampleCentroids` marks every selected sample so no
    round can draw it again (`thrust::for_each_n`,
    `kmeans_common.cuh:270-276`). Stability -- selected samples land in index
    order -- is what `DeviceSelect::If` guarantees and what this preserves.
    """
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    if flags.unsafe_load(i) != Float32(0.0):
        var pos = Int(csum.unsafe_load(i)) - 1
        out_index.unsafe_store(pos, UInt32(i))
        is_centroid.unsafe_store(i, Int32(1))


def count_labels_kernel(
    counts: MutPointer[Float32, MutAnyOrigin],
    labels: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
):
    """`countLabels` (`kmeans_common.cuh:95-135`), which is
    `cub::DeviceHistogram::HistogramEven` over the nearest-candidate labels
    with FLOAT counters -- their counter type is DataT because the result is
    consumed directly as the step-7 weight vector, and so is this.

    Float atomic adds of 1.0 are exact and commutative while the count stays
    below 2^24, which the driver's `n_samples` guard already ensures, so the
    histogram is deterministic with no ordering guarantee.
    """
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    _ = Atomic.fetch_add(
        counts.unsafe_offset(Int(labels.unsafe_load(i))), Float32(1.0)
    )


def zero_f32_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """Float sibling of `reduce_by_key.mojo::zero_i32_kernel`, for the
    step-7 weight histogram."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        a.unsafe_store(i, Float32(0.0))


def set_flag_kernel(
    flags: MutPointer[Int32, MutAnyOrigin],
    idx_in: Int32,
):
    """Mark the step-1 seed centroid.

    Theirs fills a HOST `std::vector`, sets one element and copies the whole
    vector down (`detail/kmeans.cuh:593-601`). Same values, different
    mechanism: an O(rows) host fill exists in their code because RAFT owns a
    pinned staging path; here a device zero plus this one-thread write costs
    two launches and moves nothing across the bus.
    """
    if Int(thread_idx.x) == 0 and Int(block_idx.x) == 0:
        flags.unsafe_store(Int(idx_in), Int32(1))
