"""FNV-1a32, the seed chain the whole forest's determinism hangs on.

MIRRORS `cpp/src/decisiontree/batched-levelalgo/random_utils.cuh` at
rapidsai/cuml `v26.08.00`
(`265b9da6a0e75dbef071a3168398b993a5ff6f0e`).

Their file is 52 lines and is transcribed here constant for constant and
shift for shift. It matters far out of proportion to its size, because
it is the ONLY thing standing between "this forest is reproducible" and
"this forest is reproducible on NVIDIA". Two call sites consume it:

  * `randomforest.cuh:120-123` -- the per-tree row sample seed,
    `rs = fnv1a32(fnv1a32(basis, seed), tree_id)`, fed to
    `raft::random::RngState(rs, GenPhilox)`.
  * `kernels/builder_kernels.cuh:88` -- the per-NODE feature sample
    seed, `fnv1a32_hash(seed, treeid, nodeid)`, fed to
    `cuda::std::minstd_rand`.

Note what the second one implies and their comment at
`randomforest.cuh:119` says out loud ("Hash these together so per-tree
row samples are uncorrelated"): the RNG is not a stream that threads
draw from in order, it is a PURE FUNCTION of (seed, treeid, nodeid).
There is no draw order to reproduce, so nothing here depends on launch
order, block count, warp width or stream count. This is the reason
`n_streams` could first be dropped on Metal, and now that DEVIATION 117
is PORTED it is the reason K trees can be pipelined over one queue
(`randomforest.mojo`) without touching a single output bit.

A CORRECTION, because the sentence that stood here was FALSE and a
falsified sentence gets deleted rather than annotated. It read: "their own
docs already tell users to set `n_streams=1` for reproducibility
(`randomforestclassifier.pyx:182`)". **cuML says no such thing at this
pin.** The file is `randomforestclassifier.py`, not `.pyx`; its `n_streams`
documentation is two lines at `:94-95` and reads, in full, "Number of
parallel streams used for forest building." Grepping
`reproduc|deterministic` across `python/cuml/cuml/ensemble/`,
`cpp/src/randomforest/` and `cpp/src/decisiontree/` finds no such guidance
for RF anywhere.

The conclusion survives on better evidence than a docstring, both of it
verified in their source: their own non-OpenMP build defines
`omp_get_max_threads()` to 1 (`randomforest.cuh:38-43`) and `set_rf_params`
takes `min(cfg_n_streams, omp_get_max_threads())` (`randomforest.cu:584`),
so a cuML compiled without OpenMP runs single-stream no matter what the
user passed; and both RNG draws are pure hashes of `(seed, tree_id)` and
`(seed, treeid, nodeid)`, so no output bit can depend on stream count at
all.

`fnv1a32_combine` (`random_utils.cuh:33-41`) folds a value in 32 bits at
a time, low half first, and adds the high half ONLY when
`sizeof(T) > sizeof(uint32_t)`. Their `fnv1a32_hash(seed, treeid,
nodeid)` therefore hashes `seed` (uint64 -> two rounds) then `treeid`
(int -> one round) then `nodeid` (uint32 -> one round). The width of
each argument changes the answer, so this port keeps the widths
explicit at each call site rather than promoting everything to 64 bits.

================= DEVIATION BLOCK (whole file) =================
NONE. Every value here is an exact transcription. `fnv1a32_prime`,
`fnv1a32_basis`, the four byte-extract-and-multiply rounds and the
`sizeof(T) > 4` predicate all match their file. UInt32 multiplication
wraps in Mojo exactly as `uint32_t` wraps in C++, which is the one
property this function needs.

Their variadic `fnv1a32_hash(Ts... values)` (`random_utils.cuh:43-49`)
is a C++ fold expression. Mojo's counterpart would be a variadic over
heterogeneous widths, and the widths are load-bearing (see above), so it
is spelled here as the two arities their code actually instantiates.
That is a spelling change with no value change; it is written down only
so a reader does not go looking for a variadic that was never needed.
=================================================================
"""

# `random_utils.cuh:17-18`
comptime FNV1A32_PRIME: UInt32 = 16777619
comptime FNV1A32_BASIS: UInt32 = 2166136261


@always_inline
def fnv1a32(hash: UInt32, txt: UInt32) -> UInt32:
    """`random_utils.cuh:20-31`. Four byte rounds, low byte first."""
    var h = hash
    h ^= (txt >> 0) & 0xFF
    h *= FNV1A32_PRIME
    h ^= (txt >> 8) & 0xFF
    h *= FNV1A32_PRIME
    h ^= (txt >> 16) & 0xFF
    h *= FNV1A32_PRIME
    h ^= (txt >> 24) & 0xFF
    h *= FNV1A32_PRIME
    return h


@always_inline
def fnv1a32_combine_u32(hash: UInt32, value: UInt32) -> UInt32:
    """`random_utils.cuh:33-41` with `sizeof(T) == 4`: one round."""
    return fnv1a32(hash, value)


@always_inline
def fnv1a32_combine_u64(hash: UInt32, value: UInt64) -> UInt32:
    """`random_utils.cuh:33-41` with `sizeof(T) == 8`: low half, then high.
    """
    var h = fnv1a32(hash, UInt32(value & 0xFFFFFFFF))
    return fnv1a32(h, UInt32((value >> 32) & 0xFFFFFFFF))


@always_inline
def fnv1a32_hash_seed_tree(seed: UInt64, treeid: Int32) -> UInt32:
    """`randomforest.cuh:121-122`, the per-tree row-sample seed.

    Their two lines are `rs = fnv1a32(rs, seed_); rs = fnv1a32(rs,
    tree_id);` on a `rs` initialized to `fnv1a32_basis` -- note that
    this call site uses `fnv1a32` DIRECTLY, not `fnv1a32_combine`, so
    the uint64 `seed_` is folded in ONE round on its low 32 bits and its
    high half is discarded. That asymmetry with `fnv1a32_hash` below is
    theirs; it is transcribed, not corrected.
    """
    var rs = FNV1A32_BASIS
    rs = fnv1a32(rs, UInt32(seed & 0xFFFFFFFF))
    rs = fnv1a32(rs, UInt32(Int(treeid)))
    return rs


@always_inline
def fnv1a32_hash_seed_tree_node(
    seed: UInt64, treeid: Int32, nodeid: UInt32
) -> UInt32:
    """`builder_kernels.cuh:88`, the per-node feature-sample seed.

    `fnv1a32_hash(seed, treeid, nodeid)` with the widths their call site
    supplies: `seed` is `uint64_t` (two rounds via
    `fnv1a32_combine`), `treeid` is `IdxT` = `int` (one round), `nodeid`
    is `uint32_t` (one round).
    """
    var h = FNV1A32_BASIS
    h = fnv1a32_combine_u64(h, seed)
    h = fnv1a32_combine_u32(h, UInt32(Int(treeid)))
    h = fnv1a32_combine_u32(h, nodeid)
    return h
