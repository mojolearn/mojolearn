# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The RandomForest fit on the host, a SECOND spelling of the device trainer
(workstream E batch 3, 2026-09-14; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md section 1.1 rf-clf, rf-reg).

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu`, a `DeviceContext` or
any module under `ensemble/` that defines a kernel. The only library
imports are the `checks/numerics.mojo` seams (`ftz`, `identical_mul_add`,
`identical_log`). Every other construct the device fit reaches is RESTATED
below, with the file and line of the device routine it MIRRORS, so a
disagreement between the two is a finding and not a shared bug.

WHAT IS MIRRORED, IN THE ORDER `fit_forest` REACHES IT
(`ensemble/randomforest.mojo:2299-2849`, IDENTICAL build, the default
flags: `LABELS_SAMPLED_ORDER` on, `ROWS_SORTED_SAMPLE` off,
`RETRY_PURE_NODES` off so a pure node is a leaf, `HIST_ITEMS_PER_THREAD` 4,
`HIST_SMEM_COPIES_DEFAULT` 1, the pinned split reduction)

  1. `n_bins` clamped to `n_rows` (`randomforest.mojo:2420-2426`) and
     `n_sampled_rows_for`'s Float32 round half away (`:2218-2252`).
  2. `ftz_features_kernel` (`:1260-1275`): X is flushed in place, once,
     before any reader. Every quantile, bin and partition compare below
     reads the flushed matrix.
  3. `compute_quantiles` (`batched_levelalgo/quantiles.mojo:728-925`): the
     shared row sample of `sample_owned_columns_kernel` (`:532-595`; RAFT's
     PCGenerator `:349-464` and the uint64 Lemire draw `:468-518`, the
     identity arm when `n_rows <= 4 * max_n_bins`), CUB's float twiddle
     order (`core/segmented_sort.mojo:101-105`) as an ascending sort of the
     twiddled keys, `quantile_bin_index` (`:700-725`, Float64 on the host,
     round half away), the gather and the `ftz`-compared unique of
     `compute_quantiles_batched_kernel` (`:598-660`).
  4. `bin_dataset_kernel` (`kernels/builder_kernels_impl.mojo:2971-3006`)
     with `lower_bound_aspace`'s clamped search (`:541-566`).
  5. Per tree, `RowSampler._sample_rows` (`randomforest.mojo:2093-2206`):
     the bootstrap arm is `uniform_int_kernel` (`core/philox.mojo:203-232`)
     at stride 110592, one Philox generator per output index
     (`PhiloxState` `:61-167`, `custom_next_uniform_int_u32` `:173-192`),
     seeded by `fnv1a32_hash_seed_tree`
     (`batched_levelalgo/random_utils.mojo:136-158`, DEVIATION 400's
     conditional high-half round); the no-bootstrap arm is the identity.
     The sampled-order label gather of DEVIATION 2001 is address-only and
     has no host counterpart to write: the host reads `labels[row]`.
  6. `Builder.begin_tree` / `advance_tree` (`builder.mojo:2719-2841`) over
     `NodeQueue` (`:167-433`): FIFO pops of `max_batch_size`, the
     `_is_expandable` test, `push`'s six mutations in their order.
  7. Per batch, `begin_batch` / `advance_batch` (`:2179-2320`): the
     sampling rounds (`max_sampling_rounds_for`, `sampled_cols_in_round`,
     `:143-164`), the column sample of `sampled_column_at`
     (`kernels/builder_kernels.mojo:315-335`, the per-node FNV seed
     `random_utils.mojo:162-176`, `core/shuffle_iterator.mojo:137-315`'s
     minstd keys and Feistel cycle walk), the histogram of
     `build_histograms_kernel` (`builder_kernels_impl.mojo:2149-2395`, an
     integer or fixed-point Int32 sum, so any order), `pdf_to_cdf`
     (`core/block_scan.mojo:288-350`, an integer prefix sum), the per-thread
     `Gain` of `find_best_splits_kernel` (`:2744-2922`, `objectives.mojo`
     Gini `:558-619`, Entropy `:622-700`, MSE `:922-978`), DEVIATION 2502's
     purity mark, the PINNED reduction `Split.eval_best_split_pinned`
     (`split.mojo:686-779`, width 32, phase 1 per group and phase 2 over
     the group results, each step a lockstep read then update),
     `_publish_to_global` (`:614-683`, the range midpoint then the slot's
     `update`), `_read_splits`' terminal rule (`builder.mojo:1733-1763`)
     and the retry of the invalid, non-terminal nodes.
  8. `enqueue_node_split` (`builder.mojo:2357-2452`) through
     `launch_node_split_kernel` (`builder_kernels_impl.mojo:1472-1702`):
     the local left count of `count_local_left_kernel` (`:915-966`, the
     `value(row, colid) <= quesval` test on valid splits only) and the
     scan-by-key writer's placement (`:1069-1101`), which is a stable
     partition of each valid node's range; invalid nodes keep their order.
  9. `set_leaf_predictions` (`builder.mojo:2485-2666`) through
     `leaf_kernel` (`builder_kernels_impl.mojo:1837-1952`): the class-count
     or fixed-point label histogram over the leaf's range and
     `SetLeafVector` (`objectives.mojo:816-847`, `:1356-1376`).

The Split total order (`split.mojo:432-496`), the range merge and the
midpoint rule are restated as `HostSplit` below. The regression label
plane is `RegressionBin`'s (`bins.mojo:500-600`): `Int32(label * scale)`
truncated, summed in Int32, dequantized as `Float32(raw) / scale`; the
scale is the binding's `choose_scale` over the label magnitudes, as the GPU
binding computes it (`bindings/_mojolearn_rf.mojo:594-600`).

ADDED 2026-09-15 (the forest variant lanes rf-reg-poisson, rf-reg-gamma-ig
and rf-clf-balanced-parallel): the POISSON, GAMMA and INVERSE_GAUSSIAN gains
(`objectives.mojo:981-1245`, the storage-width right label sum, the `eps_`
guards, `identical_log`), and the class-weighted BOOTSTRAP, whose weights
act only through the row draw (`randomforest.mojo:2570-2582`):
`prepare_weights`' Float64 CDF and the Philox `uniform<double>` draws read
through `upper_bound` (`:2098-2135`).

WHAT IS REFUSED BY NAME: class weights WITHOUT bootstrap (the weighted
objective, `WeightedClassificationBin`), and `max_n_bins > 1024` (their own
refusal). OOB scoring never crosses the GPU binding either.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` draws every bootstrap
row, weighted or not, from the NEXT Philox subsequence (`i + 1` in place of
`i`), so every tree of a bootstrap forest trains on a different row
multiset, and adds one to every node's column-sample seed, so a forest
without a bootstrap that samples columns moves too. An arm on an integer
fold could not fail: every histogram here is an integer sum that no order
moves.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the rf-clf and rf-reg lanes is the measurement.
"""
from std.math import ceildiv, floor
from std.memory import bitcast
from std.builtin.sort import sort
from std.sys.compile import is_defined
from max.algorithm import sync_parallelize

from checks.numerics import ftz, identical_log, identical_mul_add
from core.host_predict_threads import host_predict_chunk, host_predict_task_count
from ensemble.host_layout import RF_NAN_REFUSAL, has_nan_f32_threaded


#: The gate's negative control (see THE NEGATIVE CONTROL above).
comptime RF_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: `CRITERION` (`ensemble/decisiontree/decisiontree.mojo`, cuML's
#: `algo_helper.h:10-19`).
comptime RF_GINI = 0
comptime RF_ENTROPY = 1
comptime RF_MSE = 2
comptime RF_MAE = 3
comptime RF_POISSON = 4
comptime RF_GAMMA = 5
comptime RF_INVERSE_GAUSSIAN = 6
comptime RF_CRITERION_END = 7

#: `TPB_DEFAULT` (`builder.mojo:57`, `builder_kernels_impl.mojo:437`): the
#: block width of `find_best_splits_kernel` and so of the Gain threads.
comptime RF_TPB = 128

#: `PINNED_SPLIT_REDUCE_LANES` (`split.mojo:202`), DEVIATION 404.
comptime RF_PINNED_LANES = 32

#: `oversampling_factor`, the literal at `randomforest.mojo:2469`.
comptime RF_QUANTILE_OVERSAMPLING = 4

#: `RNG_STRIDE` (`core/philox.mojo:200`), the bootstrap launch stride.
comptime RF_RNG_STRIDE = 110592

#: `-std::numeric_limits<float>::max()`, `Split::Min()` (`split.mojo:289-300`).
comptime RF_SPLIT_MIN = Float32(-3.4028234663852886e38)


# ===========================================================================
# THE PARAMETERS, `RF_params` + `DecisionTreeParams` as the binding fills them
# (`bindings/_mojolearn_rf.mojo:192-211`).
# ===========================================================================


@fieldwise_init
struct RfHostParams(ImplicitlyCopyable, Movable):
    var n_trees: Int
    var max_depth: Int
    var max_leaves: Int
    var max_features: Float32
    var max_n_bins: Int
    var min_samples_leaf: Int
    var min_samples_split: Int
    var min_impurity_decrease: Float32
    var bootstrap: Bool
    var max_samples: Float32
    var seed: UInt64
    var n_streams: Int
    var max_batch_size: Int
    var criterion: Int


def rf_host_check_params(p: RfHostParams, classification: Bool) raises:
    """`RF_params.check` then `validity_check` (`randomforest.mojo:216-266`)
    and `DecisionTreeParams.check` (`decisiontree.mojo:248-319`), their
    messages, plus this file's refusals by name."""
    if p.criterion == RF_MAE:
        raise Error(
            "split_criterion=MAE is not supported by cuML either --"
            " `validity_check` refuses it at decisiontree.cu:28 and"
            " randomforest_common.pyx:147 raises NotImplementedError."
            " There is no upstream MAE path to implement."
        )
    if p.criterion < RF_GINI or p.criterion > RF_CRITERION_END:
        raise Error(
            "split_criterion=" + String(p.criterion)
            + " is not a CRITERION; algo_helper.h:10-18 enumerates 0"
            " (GINI) through 7 (CRITERION_END)."
        )
    if not (p.max_depth >= 0):
        raise Error("Invalid max depth " + String(p.max_depth))
    if not ((p.max_leaves == -1) or (p.max_leaves > 0)):
        raise Error("Invalid max leaves " + String(p.max_leaves))
    if not ((p.max_features > 0) and (p.max_features <= 1.0)):
        raise Error(
            "max_features value " + String(p.max_features)
            + " outside permitted (0, 1] range"
        )
    if not (p.max_n_bins > 0):
        raise Error("Invalid max_n_bins " + String(p.max_n_bins))
    if not (p.max_n_bins <= 1024):
        raise Error("max_n_bins should not be larger than 1024")
    if not (p.min_samples_leaf >= 1):
        raise Error(
            "Invalid value for min_samples_leaf " + String(p.min_samples_leaf)
            + ". Should be >= 1."
        )
    if not (p.min_samples_split >= 2):
        raise Error(
            "Invalid value for min_samples_split: " + String(p.min_samples_split)
            + ". Should be >= 2."
        )
    if not (p.n_trees > 0):
        raise Error("Invalid n_trees " + String(p.n_trees))
    if not ((p.max_samples > 0) and (p.max_samples <= 1.0)):
        raise Error(
            "max_samples value " + String(p.max_samples)
            + " outside permitted (0, 1] range"
        )
    if not (p.n_streams > 0):
        raise Error("Invalid n_streams " + String(p.n_streams))
    if p.max_batch_size < 1:
        raise Error("rf host: max_batch_size must be >= 1")
    if classification:
        if p.criterion != RF_GINI and p.criterion != RF_ENTROPY and p.criterion != RF_CRITERION_END:
            raise Error(
                "rf host: split criterion " + String(p.criterion)
                + " has no arm in the classification objective (DEVIATION 407)"
            )
    else:
        if (
            p.criterion != RF_MSE
            and p.criterion != RF_POISSON
            and p.criterion != RF_GAMMA
            and p.criterion != RF_INVERSE_GAUSSIAN
            and p.criterion != RF_CRITERION_END
        ):
            raise Error(
                "rf host: split criterion " + String(p.criterion)
                + " has no arm in the regression objective (DEVIATION 407)"
            )


# ===========================================================================
# RNG 1: RAFT's PCGenerator and the uint64 Lemire draw
# (`batched_levelalgo/quantiles.mojo:312-518`).
# ===========================================================================


@fieldwise_init
struct HostU128(ImplicitlyCopyable, Movable):
    var hi: UInt64
    var lo: UInt64


def host_wmul_64bit(a: UInt64, b: UInt64) -> HostU128:
    """`wmul_64bit`, `quantiles.mojo:313-345`: the exact 128-bit product."""
    var a_lo = a & UInt64(0xFFFFFFFF)
    var a_hi = a >> 32
    var b_lo = b & UInt64(0xFFFFFFFF)
    var b_hi = b >> 32
    var t0 = a_lo * b_lo
    var t1 = a_hi * b_lo
    var t2 = a_lo * b_hi
    var t3 = a_hi * b_hi
    var mid = (t0 >> 32) + (t1 & UInt64(0xFFFFFFFF)) + (t2 & UInt64(0xFFFFFFFF))
    var lo = (t0 & UInt64(0xFFFFFFFF)) | (mid << 32)
    var hi = t3 + (t1 >> 32) + (t2 >> 32) + (mid >> 32)
    return HostU128(hi, lo)


@fieldwise_init
struct HostPcg(ImplicitlyCopyable, Movable):
    """`PCGenerator`, `quantiles.mojo:349-464`."""

    var pcg_state: UInt64
    var inc: UInt64

    def next_u32(mut self) -> UInt32:
        """`next_u32`, `quantiles.mojo:425-450`."""
        var oldstate = self.pcg_state
        self.pcg_state = oldstate * UInt64(6364136223846793005) + self.inc
        var xorshifted = UInt32((((oldstate >> 18) ^ oldstate) >> 27) & UInt64(0xFFFFFFFF))
        var rot = UInt32((oldstate >> 59) & UInt64(0xFFFFFFFF))
        return (xorshifted >> rot) | (xorshifted << ((~rot + UInt32(1)) & UInt32(31)))

    def next_u64(mut self) -> UInt64:
        """`next_u64`, `quantiles.mojo:453-464`, low word first."""
        var a = self.next_u32()
        var b = self.next_u32()
        return UInt64(a) | (UInt64(b) << 32)

    def skipahead(mut self, offset_in: UInt64):
        """`skipahead`, `quantiles.mojo:403-422`."""
        var g = UInt64(1)
        var h = UInt64(6364136223846793005)
        var c = UInt64(0)
        var f = self.inc
        var offset = offset_in
        while offset != UInt64(0):
            if (offset & UInt64(1)) != UInt64(0):
                g = g * h
                c = c * h + f
            f = f * (h + UInt64(1))
            h = h * h
            offset >>= 1
        self.pcg_state = self.pcg_state * g + c


def host_pcg_init(seed: UInt64, subsequence: UInt64, offset: UInt64) -> HostPcg:
    """`init_pcg`, `quantiles.mojo:370-400`: two u32 warm-up draws."""
    var g = HostPcg(UInt64(0), (subsequence << 1) | UInt64(1))
    _ = g.next_u32()
    g.pcg_state += seed
    _ = g.next_u32()
    g.skipahead(offset)
    return g


def host_pcg_uniform_u64(mut gen: HostPcg, start: UInt64, diff: UInt64) -> UInt64:
    """`custom_next_uniform_int_u64`, `quantiles.mojo:468-518`."""
    var x = gen.next_u64()
    var s = diff
    var m = host_wmul_64bit(x, s)
    var m_hi = m.hi
    var m_lo = m.lo
    if m_lo < s:
        var t = (~s + UInt64(1)) % s
        while m_lo < t:
            x = gen.next_u64()
            var mm = host_wmul_64bit(x, s)
            m_hi = mm.hi
            m_lo = mm.lo
    return m_hi + start


# ===========================================================================
# RNG 2: Philox4x32-10 and `uniformInt<int>` (`core/philox.mojo`).
# ===========================================================================


def _mulhi32(a: UInt32, b: UInt32) -> UInt32:
    var p = (UInt64(a) & UInt64(0xFFFFFFFF)) * (UInt64(b) & UInt64(0xFFFFFFFF))
    return UInt32((p >> 32) & UInt64(0xFFFFFFFF))


def _mullo32(a: UInt32, b: UInt32) -> UInt32:
    var p = (UInt64(a) & UInt64(0xFFFFFFFF)) * (UInt64(b) & UInt64(0xFFFFFFFF))
    return UInt32(p & UInt64(0xFFFFFFFF))


@fieldwise_init
struct HostPhilox(ImplicitlyCopyable, Movable):
    """`PhiloxState`, `core/philox.mojo:61-167`, the four counter words, the
    two key words, the cached block and the word index as scalars."""

    var c0: UInt32
    var c1: UInt32
    var c2: UInt32
    var c3: UInt32
    var k0: UInt32
    var k1: UInt32
    var o0: UInt32
    var o1: UInt32
    var o2: UInt32
    var o3: UInt32
    var state: UInt32

    def regen(mut self):
        """`philox4x32_10`, `core/philox.mojo:27-50`: ten rounds, the key
        bumped by the Weyl constants between rounds (nine bumps)."""
        var x0 = self.c0
        var x1 = self.c1
        var x2 = self.c2
        var x3 = self.c3
        var y0 = self.k0
        var y1 = self.k1
        for r in range(10):
            # `_philox4x32_round` (`:28-36`).
            var hi0 = _mulhi32(UInt32(0xD2511F53), x0)
            var lo0 = _mullo32(UInt32(0xD2511F53), x0)
            var hi1 = _mulhi32(UInt32(0xCD9E8D57), x2)
            var lo1 = _mullo32(UInt32(0xCD9E8D57), x2)
            var n0 = hi1 ^ x1 ^ y0
            var n1 = lo1
            var n2 = hi0 ^ x3 ^ y1
            var n3 = lo0
            x0 = n0
            x1 = n1
            x2 = n2
            x3 = n3
            if r < 9:
                y0 = y0 + UInt32(0x9E3779B9)
                y1 = y1 + UInt32(0xBB67AE85)
        self.o0 = x0
        self.o1 = x1
        self.o2 = x2
        self.o3 = x3

    def incr(mut self):
        """`_incr`, `core/philox.mojo:86-96`."""
        self.c0 = self.c0 + UInt32(1)
        if self.c0 == UInt32(0):
            self.c1 = self.c1 + UInt32(1)
            if self.c1 == UInt32(0):
                self.c2 = self.c2 + UInt32(1)
                if self.c2 == UInt32(0):
                    self.c3 = self.c3 + UInt32(1)

    def incr_n(mut self, n: UInt64):
        """`_incr_n`, `core/philox.mojo:99-112`; the carry test is `nhi <= c1`."""
        var nlo = UInt32(n & UInt64(0xFFFFFFFF))
        var nhi = UInt32((n >> 32) & UInt64(0xFFFFFFFF))
        self.c0 = self.c0 + nlo
        if self.c0 < nlo:
            nhi = nhi + UInt32(1)
        self.c1 = self.c1 + nhi
        if not (nhi <= self.c1):
            self.c2 = self.c2 + UInt32(1)
            if self.c2 == UInt32(0):
                self.c3 = self.c3 + UInt32(1)

    def incr_hi(mut self, n: UInt64):
        """`_incr_hi`, `core/philox.mojo:115-124`."""
        var nlo = UInt32(n & UInt64(0xFFFFFFFF))
        var nhi = UInt32((n >> 32) & UInt64(0xFFFFFFFF))
        self.c2 = self.c2 + nlo
        if self.c2 < nlo:
            nhi = nhi + UInt32(1)
        self.c3 = self.c3 + nhi

    def skipahead(mut self, n_in: UInt64):
        """`skipahead`, `core/philox.mojo:138-147`."""
        var n = n_in
        self.state = self.state + UInt32(n & UInt64(3))
        n = n // UInt64(4)
        if self.state > UInt32(3):
            n = n + UInt64(1)
            self.state = self.state - UInt32(4)
        self.incr_n(n)
        self.regen()

    def next_u32(mut self) -> UInt32:
        """`next_u32`, `core/philox.mojo:150-167`."""
        var s = self.state
        self.state = s + UInt32(1)
        var ret: UInt32
        if s == UInt32(1):
            ret = self.o1
        elif s == UInt32(2):
            ret = self.o2
        elif s == UInt32(3):
            ret = self.o3
        else:
            ret = self.o0
        if self.state == UInt32(4):
            self.incr()
            self.regen()
            self.state = UInt32(0)
        return ret


def host_philox_init(seed: UInt64, subsequence: UInt64, offset: UInt64) -> HostPhilox:
    """`PhiloxState.init`, `core/philox.mojo:71-83`."""
    var s = HostPhilox(
        UInt32(0), UInt32(0), UInt32(0), UInt32(0),
        UInt32(seed & UInt64(0xFFFFFFFF)), UInt32((seed >> 32) & UInt64(0xFFFFFFFF)),
        UInt32(0), UInt32(0), UInt32(0), UInt32(0), UInt32(0),
    )
    # `skipahead_sequence` (`:132-135`).
    s.incr_hi(subsequence)
    s.regen()
    s.skipahead(offset)
    return s


def host_philox_uniform_int(mut gen: HostPhilox, start: Int32, diff: UInt32) -> Int32:
    """`custom_next_uniform_int_u32`, `core/philox.mojo:173-192`."""
    var s = diff
    var x = gen.next_u32()
    var m = (UInt64(x) & UInt64(0xFFFFFFFF)) * (UInt64(s) & UInt64(0xFFFFFFFF))
    var l = UInt32(m & UInt64(0xFFFFFFFF))
    if l < s:
        var t = (~s + UInt32(1)) % s
        while l < t:
            x = gen.next_u32()
            m = (UInt64(x) & UInt64(0xFFFFFFFF)) * (UInt64(s) & UInt64(0xFFFFFFFF))
            l = UInt32(m & UInt64(0xFFFFFFFF))
    var hi = UInt32((m >> 32) & UInt64(0xFFFFFFFF))
    return (hi + start.cast[DType.uint32]()).cast[DType.int32]()


# ===========================================================================
# RNG 3: FNV-1a seeds (`batched_levelalgo/random_utils.mojo:101-176`) and the
# feature shuffle (`core/shuffle_iterator.mojo:102-315`).
# ===========================================================================


def host_fnv1a32(hash: UInt32, txt: UInt32) -> UInt32:
    """`fnv1a32`, `random_utils.mojo:107-118`, four byte rounds low byte first."""
    var h = hash
    h ^= (txt >> 0) & UInt32(0xFF)
    h *= UInt32(16777619)
    h ^= (txt >> 8) & UInt32(0xFF)
    h *= UInt32(16777619)
    h ^= (txt >> 16) & UInt32(0xFF)
    h *= UInt32(16777619)
    h ^= (txt >> 24) & UInt32(0xFF)
    h *= UInt32(16777619)
    return h


def host_seed_tree(seed: UInt64, treeid: Int) -> UInt32:
    """`fnv1a32_hash_seed_tree`, `random_utils.mojo:136-158`, with DEVIATION
    400's high-half round exactly when the high half is nonzero."""
    var rs = UInt32(2166136261)
    rs = host_fnv1a32(rs, UInt32(seed & UInt64(0xFFFFFFFF)))
    var hi = UInt32((seed >> 32) & UInt64(0xFFFFFFFF))
    if hi != UInt32(0):
        rs = host_fnv1a32(rs, hi)
    rs = host_fnv1a32(rs, UInt32(treeid))
    return rs


def host_seed_tree_node(seed: UInt64, treeid: Int, nodeid: Int) -> UInt32:
    """`fnv1a32_hash_seed_tree_node`, `random_utils.mojo:162-176`: the uint64
    seed in two rounds (low, then high), the tree id, the node's TREE index."""
    var h = UInt32(2166136261)
    h = host_fnv1a32(h, UInt32(seed & UInt64(0xFFFFFFFF)))
    h = host_fnv1a32(h, UInt32((seed >> 32) & UInt64(0xFFFFFFFF)))
    h = host_fnv1a32(h, UInt32(treeid))
    h = host_fnv1a32(h, UInt32(nodeid))
    comptime if RF_ORACLE_HOST_SABOTAGE:
        # THE NEGATIVE CONTROL's second arm: every node's column-sample seed
        # one off, so a forest without a bootstrap draw moves too.
        h = h + UInt32(1)
    return h


def _host_lcg_next(mut x: UInt64) -> UInt64:
    """`lcg_next`, `shuffle_iterator.mojo:153-162`: advance, then return."""
    x = (UInt64(48271) * x) % UInt64(2147483647)
    return x


struct HostFeistel(Movable):
    """`FeistelBijection`, `core/shuffle_iterator.mojo:195-295`, built once per
    node. The device rebuilds it per sampled index; the keys and the walk
    are a pure function of `(num_elements, seed)`, so the values agree."""

    var num_elements: UInt64
    var left_bits: UInt64
    var right_bits: UInt64
    var left_mask: UInt64
    var right_mask: UInt64
    var keys: List[UInt32]

    def __init__(out self, num_elements: Int, seed: UInt32):
        self.num_elements = UInt64(max(1, num_elements))
        var max_index = self.num_elements - UInt64(1)
        # `bit_width(max_index)` with the `max(8, ...)` floor (`:232-239`).
        var width = 0
        var probe = max_index
        while probe != UInt64(0):
            width += 1
            probe >>= 1
        var total_bits = UInt64(max(8, width))
        self.left_bits = total_bits // UInt64(2)
        self.right_bits = total_bits - self.left_bits
        self.left_mask = (UInt64(1) << self.left_bits) - UInt64(1)
        self.right_mask = (UInt64(1) << self.right_bits) - UInt64(1)
        # `lcg_seed` (`:137-149`): `s % M`, a zero state rescued to 1.
        var x = UInt64(Int(seed)) % UInt64(2147483647)
        if x == UInt64(0):
            x = UInt64(1)
        self.keys = List[UInt32](capacity=24)
        for _ in range(24):
            # `key_stream_next` (`:166-191`): two rejection-tested draws,
            # high half first.
            var sp = UInt64(0)
            for _ in range(2):
                var u = _host_lcg_next(x) - UInt64(1)
                while u >= UInt64(2147418112):
                    u = _host_lcg_next(x) - UInt64(1)
                sp = ((sp << 16) + (u & UInt64(0xFFFF))) & UInt64(0xFFFFFFFF)
            self.keys.append(UInt32(Int(sp)))

    def _round_trip(self, val: UInt64) -> UInt64:
        """`_round_trip`, `shuffle_iterator.mojo:247-278`, 24 rounds, the
        32-bit shifts kept."""
        var l = (val >> self.right_bits) & UInt64(0xFFFFFFFF)
        var r = val & self.right_mask
        for i in range(24):
            var product = UInt64(0xD2B74407B1CE6E93) * l
            var f_k = ((product >> 32) & UInt64(0xFFFFFFFF)) ^ UInt64(Int(self.keys[i]))
            var b_k = product & UInt64(0xFFFFFFFF)
            var l_prime = f_k ^ r
            var r_prime = (
                (b_k << (self.right_bits - self.left_bits)) & UInt64(0xFFFFFFFF)
            ) | (r >> self.left_bits)
            l = l_prime & self.left_mask
            r = r_prime & self.right_mask
        return (l << self.right_bits) | r

    def at(self, index: Int) -> Int:
        """`__call__`, `shuffle_iterator.mojo:280-295`: a do-while cycle walk."""
        var n = UInt64(index)
        while True:
            n = self._round_trip(n)
            if n < self.num_elements:
                break
        return Int(n)


# ===========================================================================
# THE SPLIT CANDIDATE AND ITS TOTAL ORDER (`split.mojo:243-496`).
# ===========================================================================


@fieldwise_init
struct HostSplit(ImplicitlyCopyable, Movable):
    var pure: Int32
    var quesval: Float32
    var colid: Int32
    var best_metric_val: Float32
    var global_n_left: Int64
    var local_n_left: Int64
    var split_start: Int32
    var split_end: Int32

    @staticmethod
    def empty() -> HostSplit:
        """`Split()`, `split.mojo:302-312`."""
        return HostSplit(
            Int32(0), RF_SPLIT_MIN, Int32(-1), RF_SPLIT_MIN,
            Int64(0), Int64(0), Int32(-1), Int32(-1),
        )

    def is_valid(self) -> Bool:
        return self.colid != Int32(-1)

    def has_valid_split_range(self) -> Bool:
        """`split.mojo:339-342`."""
        return self.split_start >= Int32(0) and self.split_end >= self.split_start

    def replace_with(mut self, o: HostSplit) -> Bool:
        """`replace_with`, `split.mojo:383-402`; `local_nLeft` reset to 0."""
        self.quesval = o.quesval
        self.colid = o.colid
        self.best_metric_val = o.best_metric_val
        self.global_n_left = o.global_n_left
        self.local_n_left = 0
        self.split_start = o.split_start
        self.split_end = o.split_end
        return True

    def update(mut self, o: HostSplit) -> Bool:
        """`update`, `split.mojo:431-496`, branch for branch."""
        if o.best_metric_val > self.best_metric_val:
            return self.replace_with(o)
        if o.best_metric_val != self.best_metric_val:
            return False
        if o.colid > self.colid:
            return self.replace_with(o)
        if o.colid != self.colid:
            return False
        # `can_merge_equivalent_split_range` (`:344-357`).
        if (
            self.global_n_left == o.global_n_left
            and self.has_valid_split_range()
            and o.split_start >= Int32(0)
            and o.split_end >= o.split_start
        ):
            # `merge_equivalent_split_range` (`:359-381`).
            if o.split_start < self.split_start:
                self.split_start = o.split_start
            if o.split_end > self.split_end:
                self.split_end = o.split_end
            if o.quesval > self.quesval:
                self.quesval = o.quesval
            return True
        if o.quesval > self.quesval:
            return self.replace_with(o)
        return False


def host_pinned_reduce(mut threads: List[HostSplit], tpb: Int) raises -> HostSplit:
    """`eval_best_split_pinned`, `split.mojo:686-779`, for one block of `tpb`
    threads, returning thread 0's value before the publish.

    Phase 1 is the width-32 rotate-and-reduce per group of 32 threads: each
    step every thread reads its neighbor's PRE-step value (the read sits
    before a barrier, the update and store after it), so the step is a
    synchronous update. Phase 2 seeds group 0's lanes with the group
    results (lanes past the group count seed `Split()`) and runs the same
    steps over those 32 lanes."""
    comptime L = RF_PINNED_LANES
    if tpb % L != 0:
        raise Error("rf host: the pinned split reduction needs TPB % 32 == 0 (DEVIATION 404)")
    var n_groups = tpb // L
    for g in range(n_groups):
        var off = L // 2
        while off >= 1:
            var snap = List[HostSplit](capacity=L)
            for v in range(L):
                snap.append(threads[g * L + v])
            for v in range(L):
                var cur = threads[g * L + v]
                _ = cur.update(snap[(v + off) % L])
                threads[g * L + v] = cur
            off = off // 2
    var lanes = List[HostSplit](capacity=L)
    for v in range(L):
        if v < n_groups:
            lanes.append(threads[v * L])
        else:
            lanes.append(HostSplit.empty())
    var off2 = L // 2
    while off2 >= 1:
        var snap2 = List[HostSplit](capacity=L)
        for v in range(L):
            snap2.append(lanes[v])
        for v in range(L):
            var cur2 = lanes[v]
            _ = cur2.update(snap2[(v + off2) % L])
            lanes[v] = cur2
        off2 = off2 // 2
    return lanes[0]


# ===========================================================================
# THE GAINS (`objectives.mojo`), Float32 through the IDENTICAL seams.
# ===========================================================================


def _cls_weight_at(hist: List[UInt32], i: Int, n_bins: Int, n_classes: Int) -> Int64:
    """`WeightAt`, `objectives.mojo:533-555`, the exact Int64 count."""
    var weight = Int64(0)
    for j in range(n_classes):
        weight = weight + Int64(Int(hist[n_bins * j + i]))
    return weight


def host_gini_gain(hist: List[UInt32], i: Int, n_bins: Int, n_classes: Int) -> Float32:
    """`GiniGain`, `objectives.mojo:557-619`."""
    var one = Float32(1.0)
    var total_weight = _cls_weight_at(hist, n_bins - 1, n_bins, n_classes)
    var left_weight = _cls_weight_at(hist, i, n_bins, n_classes)
    var right_weight = total_weight - left_weight
    if total_weight <= 0 or left_weight <= 0 or right_weight <= 0:
        return RF_SPLIT_MIN
    var inv_len = ftz(one / total_weight.cast[DType.float32]())
    var inv_left = ftz(one / left_weight.cast[DType.float32]())
    var inv_right = ftz(one / right_weight.cast[DType.float32]())
    var gain = Float32(0.0)
    for j in range(n_classes):
        var val_i = Int64(0)
        var lval_i = Int64(Int(hist[n_bins * j + i]))
        var lval = ftz(lval_i.cast[DType.float32]())
        var l1 = ftz(lval * inv_left)
        var l2 = ftz(l1 * lval)
        gain = identical_mul_add(l2, inv_len, gain)
        val_i += lval_i
        var total_sum = Int64(Int(hist[n_bins * j + n_bins - 1]))
        var rval_i = total_sum - lval_i
        var rval = ftz(rval_i.cast[DType.float32]())
        var r1 = ftz(rval * inv_right)
        var r2 = ftz(r1 * rval)
        gain = identical_mul_add(r2, inv_len, gain)
        val_i += rval_i
        var val = ftz(val_i.cast[DType.float32]() * inv_len)
        gain = identical_mul_add(-val, val, gain)
    return gain


def host_entropy_gain(hist: List[UInt32], i: Int, n_bins: Int, n_classes: Int) -> Float32:
    """`EntropyGain`, `objectives.mojo:621-700`; `raft::log(2)` recomputed per
    term where they compute it."""
    var total_weight = _cls_weight_at(hist, n_bins - 1, n_bins, n_classes)
    var left_weight = _cls_weight_at(hist, i, n_bins, n_classes)
    var right_weight = total_weight - left_weight
    if total_weight <= 0 or left_weight <= 0 or right_weight <= 0:
        return RF_SPLIT_MIN
    var gain = Float32(0.0)
    var inv_left = ftz(Float32(1.0) / left_weight.cast[DType.float32]())
    var inv_right = ftz(Float32(1.0) / right_weight.cast[DType.float32]())
    var inv_len = ftz(Float32(1.0) / total_weight.cast[DType.float32]())
    for c in range(n_classes):
        var val_i = Int64(0)
        var lval_i = Int64(Int(hist[n_bins * c + i]))
        if lval_i != 0:
            var lval = ftz(lval_i.cast[DType.float32]())
            var larg = ftz(lval * inv_left)
            var l1 = ftz(identical_log(larg) / identical_log(Float32(2)))
            var l2 = ftz(l1 * lval)
            gain = identical_mul_add(l2, inv_len, gain)
        val_i += lval_i
        var total_sum = Int64(Int(hist[n_bins * c + n_bins - 1]))
        var rval_i = total_sum - lval_i
        if rval_i != 0:
            var rval = ftz(rval_i.cast[DType.float32]())
            var rarg = ftz(rval * inv_right)
            var r1 = ftz(identical_log(rarg) / identical_log(Float32(2)))
            var r2 = ftz(r1 * rval)
            gain = identical_mul_add(r2, inv_len, gain)
        val_i += rval_i
        if val_i != 0:
            var val = ftz(val_i.cast[DType.float32]() * inv_len)
            var v1 = ftz(val * identical_log(val))
            var v2 = ftz(v1 / identical_log(Float32(2)))
            gain = ftz(gain - v2)
    return gain


def _dequantize(raw: Int32, scale: Float32) -> Float32:
    """`_dequantize`, `bins.mojo:193-197`."""
    return Float32(Int(raw)) / scale


def _dequantize_wide(raw: Int64, scale: Float32) -> Float32:
    """`_dequantize_wide`, `bins.mojo:183-189`: one rounding after the
    storage-width subtraction."""
    return Float32(Int(raw)) / scale


comptime RF_REG_EPS = Float32(10.0) * Float32(1.1920928955078125e-07)
"""`RegressionObjectiveFunction.eps_`, `objectives.mojo:895`:
`10 * numeric_limits<float>::epsilon()`, the epsilon written as 2^-23."""


def host_poisson_gain(
    counts: List[UInt32], label_sums: List[Int32], i: Int, n_bins: Int, scale: Float32
) -> Float32:
    """`PoissonGain`, `objectives.mojo:981-1064`: the right label sum
    subtracted at storage width (`LabelSumMinus`, `bins.mojo`), the `eps_`
    guards, every `raft::log` through `identical_log` (`core/tree_math.mojo`
    for Float32), each store flushed."""
    var parent_weight = Int64(Int(counts[n_bins - 1]))
    var left_weight = Int64(Int(counts[i]))
    var right_weight = parent_weight - left_weight
    if parent_weight <= 0 or left_weight <= 0 or right_weight <= 0:
        return RF_SPLIT_MIN
    var inv_len = ftz(Float32(1) / parent_weight.cast[DType.float32]())
    var label_sum = ftz(_dequantize(label_sums[n_bins - 1], scale))
    var left_label_sum = ftz(_dequantize(label_sums[i], scale))
    var right_label_sum = ftz(_dequantize_wide(
        Int64(Int(label_sums[n_bins - 1])) - Int64(Int(label_sums[i])), scale
    ))
    if label_sum <= RF_REG_EPS or left_label_sum <= RF_REG_EPS or right_label_sum <= RF_REG_EPS:
        return RF_SPLIT_MIN
    var parg = ftz(label_sum * inv_len)
    var parent_obj = ftz(-label_sum * identical_log(parg))
    var larg = ftz(left_label_sum / left_weight.cast[DType.float32]())
    var left_obj = ftz(-left_label_sum * identical_log(larg))
    var rarg = ftz(right_label_sum / right_weight.cast[DType.float32]())
    var right_obj = ftz(-right_label_sum * identical_log(rarg))
    var lr = ftz(left_obj + right_obj)
    var gain = ftz(parent_obj - lr)
    gain = ftz(gain * inv_len)
    return gain


def host_gamma_gain(
    counts: List[UInt32], label_sums: List[Int32], i: Int, n_bins: Int, scale: Float32
) -> Float32:
    """`GammaGain`, `objectives.mojo:1067-1156`, `host_poisson_gain`'s shape
    with the weights as the log factors."""
    var parent_weight = Int64(Int(counts[n_bins - 1]))
    var left_weight = Int64(Int(counts[i]))
    var right_weight = parent_weight - left_weight
    if parent_weight <= 0 or left_weight <= 0 or right_weight <= 0:
        return RF_SPLIT_MIN
    var inv_len = ftz(Float32(1) / parent_weight.cast[DType.float32]())
    var label_sum = ftz(_dequantize(label_sums[n_bins - 1], scale))
    var left_label_sum = ftz(_dequantize(label_sums[i], scale))
    var right_label_sum = ftz(_dequantize_wide(
        Int64(Int(label_sums[n_bins - 1])) - Int64(Int(label_sums[i])), scale
    ))
    if label_sum <= RF_REG_EPS or left_label_sum <= RF_REG_EPS or right_label_sum <= RF_REG_EPS:
        return RF_SPLIT_MIN
    var parg = ftz(label_sum * inv_len)
    var parent_obj = ftz(parent_weight.cast[DType.float32]() * identical_log(parg))
    var larg = ftz(left_label_sum / left_weight.cast[DType.float32]())
    var left_obj = ftz(left_weight.cast[DType.float32]() * identical_log(larg))
    var rarg = ftz(right_label_sum / right_weight.cast[DType.float32]())
    var right_obj = ftz(right_weight.cast[DType.float32]() * identical_log(rarg))
    var lr = ftz(left_obj + right_obj)
    var gain = ftz(parent_obj - lr)
    gain = ftz(gain * inv_len)
    return gain


def host_inverse_gaussian_gain(
    counts: List[UInt32], label_sums: List[Int32], i: Int, n_bins: Int, scale: Float32
) -> Float32:
    """`InverseGaussianGain`, `objectives.mojo:1159-1245`: no transcendental."""
    var parent_weight = Int64(Int(counts[n_bins - 1]))
    var left_weight = Int64(Int(counts[i]))
    var right_weight = parent_weight - left_weight
    if parent_weight <= 0 or left_weight <= 0 or right_weight <= 0:
        return RF_SPLIT_MIN
    var label_sum = ftz(_dequantize(label_sums[n_bins - 1], scale))
    var left_label_sum = ftz(_dequantize(label_sums[i], scale))
    var right_label_sum = ftz(_dequantize_wide(
        Int64(Int(label_sums[n_bins - 1])) - Int64(Int(label_sums[i])), scale
    ))
    if label_sum <= RF_REG_EPS or left_label_sum <= RF_REG_EPS or right_label_sum <= RF_REG_EPS:
        return RF_SPLIT_MIN
    var pw = parent_weight.cast[DType.float32]()
    var lw = left_weight.cast[DType.float32]()
    var rw = right_weight.cast[DType.float32]()
    var psq = ftz(-pw * pw)
    var parent_obj = ftz(psq / label_sum)
    var lsq = ftz(-lw * lw)
    var left_obj = ftz(lsq / left_label_sum)
    var rsq = ftz(-rw * rw)
    var right_obj = ftz(rsq / right_label_sum)
    var lr = ftz(left_obj + right_obj)
    var gain = ftz(parent_obj - lr)
    var denom = ftz(Float32(2) * pw)
    gain = ftz(gain / denom)
    return gain


def host_mse_gain(
    counts: List[UInt32], label_sums: List[Int32], i: Int, n_bins: Int, scale: Float32
) -> Float32:
    """`MSEGain`, `objectives.mojo:921-978`, over `RegressionBin`'s count and
    fixed-point label planes."""
    var parent_weight = Int64(Int(counts[n_bins - 1]))
    var left_weight = Int64(Int(counts[i]))
    var right_weight = parent_weight - left_weight
    if parent_weight <= 0 or left_weight <= 0 or right_weight <= 0:
        return RF_SPLIT_MIN
    var inv_len = ftz(Float32(1.0) / parent_weight.cast[DType.float32]())
    var label_sum = ftz(_dequantize(label_sums[n_bins - 1], scale))
    var left_label_sum = ftz(_dequantize(label_sums[i], scale))
    var p1 = ftz(-label_sum * label_sum)
    var parent_obj = ftz(p1 * inv_len)
    var lsq = ftz(left_label_sum * left_label_sum)
    var left_obj = ftz((-lsq) / left_weight.cast[DType.float32]())
    var right_label_sum = ftz(label_sum - left_label_sum)
    var rsq = ftz(right_label_sum * right_label_sum)
    var right_obj = ftz((-rsq) / right_weight.cast[DType.float32]())
    var lr = ftz(left_obj + right_obj)
    var gain = ftz(parent_obj - lr)
    var half_inv = ftz(Float32(0.5) * inv_len)
    gain = ftz(gain * half_inv)
    return gain


# ===========================================================================
# THE QUANTILES AND THE BINNED MATRIX, once per forest.
# ===========================================================================


struct RfHostQuantiles(Movable):
    var values: List[Float32]
    var n_bins: List[Int32]
    var bins: List[Int32]

    def __init__(out self):
        self.values = List[Float32]()
        self.n_bins = List[Int32]()
        self.bins = List[Int32]()


def _float_to_sortable(bits: UInt32) -> UInt32:
    """`float_to_sortable`, `core/segmented_sort.mojo:101-105` (CUB TwiddleIn)."""
    if (bits & UInt32(0x80000000)) != UInt32(0):
        return ~bits
    return bits | UInt32(0x80000000)


def _sortable_to_float(key: UInt32) -> UInt32:
    """`sortable_to_float`, `core/segmented_sort.mojo:109-113` (TwiddleOut)."""
    if (key & UInt32(0x80000000)) != UInt32(0):
        return key & UInt32(0x7FFFFFFF)
    return ~key


def host_quantile_bin_index(bin: Int, sample_count: Int, max_n_bins: Int) -> Int:
    """`quantile_bin_index`, `quantiles.mojo:700-725`, Float64 round half away."""
    var bin_width = Float64(sample_count) / Float64(max_n_bins)
    var x = Float64(bin + 1) * bin_width
    var r = floor(x)
    if x - r >= Float64(0.5):
        r = r + Float64(1.0)
    var idx = Int(r) - 1
    if idx < 0:
        idx = 0
    if idx > sample_count - 1:
        idx = sample_count - 1
    return idx


def host_lower_bound(values: List[Float32], base: Int, n: Int, element: Float32) -> Int:
    """`lower_bound_aspace`, `builder_kernels_impl.mojo:541-566`: the search
    runs over `[0, n - 1]`, which clamps a value past the last quantile to
    `n - 1`."""
    var start = 0
    var end = n - 1
    while start < end:
        var mid = (start + end) // 2
        if values[base + mid] < element:
            start = mid + 1
        else:
            end = mid
    return start


def host_compute_quantiles(
    x: List[Float32], n_rows: Int, n_cols: Int, max_n_bins: Int, seed: UInt64
) raises -> RfHostQuantiles:
    """`compute_quantiles` (`quantiles.mojo:728-925`) then `bin_dataset_kernel`
    (`builder_kernels_impl.mojo:2971-3006`). `x` is the FLUSHED column-major
    matrix."""
    if max_n_bins <= 0:
        raise Error("max_n_bins must be positive")
    if n_rows <= 0:
        raise Error("n_rows must be positive")
    if n_cols <= 0:
        raise Error("n_cols must be positive")
    var global_rows = UInt64(n_rows)
    var budget = UInt64(max_n_bins) * UInt64(RF_QUANTILE_OVERSAMPLING)
    var sample_count = Int(global_rows if global_rows < budget else budget)
    var out = RfHostQuantiles()

    # `sample_owned_columns_kernel` (`:532-595`): ONE row per sample index,
    # shared by every column; the identity when the budget covers the data.
    var sample_rows = List[Int](capacity=sample_count)
    for sample_idx in range(sample_count):
        var global_row = UInt64(sample_idx)
        if UInt64(sample_count) != global_rows:
            var gen = host_pcg_init(seed, UInt64(sample_idx), UInt64(0))
            global_row = host_pcg_uniform_u64(gen, UInt64(0), global_rows)
        sample_rows.append(Int(global_row))

    var bin_idx = List[Int](capacity=max_n_bins)
    for b in range(max_n_bins):
        bin_idx.append(host_quantile_bin_index(b, sample_count, max_n_bins))

    out.values = List[Float32](length=n_cols * max_n_bins, fill=Float32(0.0))
    out.n_bins = List[Int32](length=n_cols, fill=Int32(0))
    for col in range(n_cols):
        # The segmented radix sort (`core/segmented_sort.mojo`) over CUB's
        # twiddled keys: an ascending sort of the keys, which moves bits
        # and sums nothing.  The production host path may use the standard
        # integer sort here: the keys already encode the complete float
        # total order, and equal keys are identical bits, so stability is
        # unobservable while avoiding quadratic quantile preprocessing.
        var keys = List[UInt32](capacity=sample_count)
        for s in range(sample_count):
            keys.append(_float_to_sortable(bitcast[DType.uint32](x[col * n_rows + sample_rows[s]])))
        sort(keys)
        var col_q = col * max_n_bins
        # `compute_quantiles_batched_kernel` (`:598-660`): the gather, then
        # the sequential unique over `ftz`-flushed operands (DEVIATION 403).
        for b in range(max_n_bins):
            out.values[col_q + b] = bitcast[DType.float32](_sortable_to_float(keys[bin_idx[b]]))
        var w = 1
        for r in range(1, max_n_bins):
            var prev = out.values[col_q + w - 1]
            var cur = out.values[col_q + r]
            if ftz(cur) != ftz(prev):
                out.values[col_q + w] = cur
                w += 1
        out.n_bins[col] = Int32(w)

    # DEVIATION 314's binned matrix, the same index `lower_bound` returns.
    var bins = List[Int32](length=n_rows * n_cols, fill=Int32(0))
    var bin_tasks = host_predict_task_count(n_cols)
    if n_rows * n_cols < (1 << 19):
        bin_tasks = 1
    var bin_chunk = host_predict_chunk(n_cols, bin_tasks)
    def _bin_columns(task: Int) {imm x, imm out, mut bins, imm n_rows, imm n_cols, imm max_n_bins, imm bin_chunk}:
        var lo = task * bin_chunk
        var hi = min(lo + bin_chunk, n_cols)
        for col in range(lo, hi):
            var nb = Int(out.n_bins[col])
            for row in range(n_rows):
                var b = host_lower_bound(
                    out.values, col * max_n_bins, nb, x[col * n_rows + row]
                )
                bins[col * n_rows + row] = Int32(b)
    if bin_tasks == 1:
        _bin_columns(0)
    else:
        sync_parallelize(_bin_columns, bin_tasks)
    out.bins = bins^
    return out^


# ===========================================================================
# THE FOREST.
# ===========================================================================


struct RfHostForest(Movable):
    """The five model arrays `_forest_out` and `forest_export` publish
    (`bindings/_mojolearn_rf.mojo:214-316`), flat, plus the counts."""

    var offsets: List[Int32]
    var colid: List[Int32]
    var quesval: List[Float32]
    var left_child: List[Int32]
    var leaves: List[Float32]
    var n_trees: Int
    var num_outputs: Int

    def __init__(out self):
        self.offsets = List[Int32]()
        self.colid = List[Int32]()
        self.quesval = List[Float32]()
        self.left_child = List[Int32]()
        self.leaves = List[Float32]()
        self.n_trees = 0
        self.num_outputs = 1

    def n_nodes(self) -> Int:
        return len(self.colid)


@fieldwise_init
struct HostWorkItem(ImplicitlyCopyable, Movable):
    """`NodeWorkItem` (`kernels/builder_kernels.mojo:141-157`): the node's TREE
    index, its depth and its instance range as it stood when pushed."""

    var idx: Int
    var depth: Int
    var begin: Int
    var count: Int


def n_sampled_rows_host(bootstrap: Bool, max_samples: Float32, n_rows: Int) -> Int:
    """`n_sampled_rows_for`, `randomforest.mojo:2218-2252`."""
    if not bootstrap:
        return n_rows
    var x = Float32(max_samples) * Float32(n_rows)
    var f = Float32(Int(x))
    if x - f >= 0.5:
        return Int(f) + 1
    return Int(f)


def host_philox_uniform_double(mut gen: HostPhilox, start: Float64, end: Float64) -> Float64:
    """`custom_next_uniform_double`, `core/philox.mojo:349-354`: `next_u64`
    low word first, the top 53 bits over 2^53, times the span, plus the
    start (their order)."""
    var a = UInt64(Int(gen.next_u32())) & UInt64(0xFFFFFFFF)
    var b = UInt64(Int(gen.next_u32())) & UInt64(0xFFFFFFFF)
    var v = (a | (b << 32)) >> 11
    var res = Float64(Int(v)) / Float64(Int(UInt64(1) << 53))
    return (res * (end - start)) + start


def host_weight_cdf(weights: List[Float32], n_rows: Int) raises -> List[Float64]:
    """`RowSampler.prepare_weights`, `randomforest.mojo:1931-2000`: their two
    refusals by value, then the inclusive Float64 scan whose LAST element is
    the draw span."""
    if len(weights) < n_rows:
        raise Error(
            "sample_weight holds " + String(len(weights)) + " values but n_rows is "
            + String(n_rows)
        )
    var total = Float64(0.0)
    for i in range(n_rows):
        var w = weights[i]
        if not (w == w):
            raise Error(
                "sample_weight values must be finite and non-negative; index "
                + String(i) + " is NaN"
            )
        if w < Float32(0.0):
            raise Error(
                "sample_weight values must be finite and non-negative; index "
                + String(i) + " is " + String(w)
            )
        total += Float64(w)
    if total <= 0.0:
        raise Error(
            "sample_weight values must contain at least one positive value"
            " (randomforest.cuh:93-95)"
        )
    var cdf = List[Float64](capacity=n_rows)
    var run = Float64(0.0)
    for i in range(n_rows):
        run += Float64(weights[i])
        cdf.append(run)
    return cdf^


def host_sampled_rows(
    seed: UInt64,
    tree_id: Int,
    bootstrap: Bool,
    n_rows: Int,
    n_sampled: Int,
    weight_cdf: List[Float64] = List[Float64](),
) raises -> List[Int32]:
    """`RowSampler._sample_rows`, `randomforest.mojo:2093-2206`: the two
    unweighted arms and the weighted bootstrap (`weight_cdf` non-empty)."""
    var rows = List[Int32](capacity=n_sampled)
    if bootstrap and len(weight_cdf) > 0:
        # `:2098-2135`: `uniform_double_host` over `[0, weight_sum)` at stride
        # 110592 (`core/philox.mojo:357-379`, one generator per subsequence),
        # then `std::upper_bound` over the CDF.
        var weight_sum = weight_cdf[n_rows - 1]
        var draw_seed = UInt64(Int(host_seed_tree(seed, tree_id)))
        for i in range(n_sampled):
            var sub = UInt64(i % RF_RNG_STRIDE)
            comptime if RF_ORACLE_HOST_SABOTAGE:
                sub = sub + UInt64(1)
            var gen = host_philox_init(draw_seed, sub, UInt64(0))
            for _ in range(i // RF_RNG_STRIDE):
                _ = host_philox_uniform_double(gen, Float64(0.0), weight_sum)
            var d = host_philox_uniform_double(gen, Float64(0.0), weight_sum)
            var lo = 0
            var hi = n_rows
            while lo < hi:
                var mid = (lo + hi) // 2
                if weight_cdf[mid] <= d:
                    lo = mid + 1
                else:
                    hi = mid
            rows.append(Int32(lo))
        return rows^
    if not bootstrap:
        # DEVIATION 2484's sequence, `row_ids_tiled_sequence_kernel`.
        for i in range(n_sampled):
            rows.append(Int32(i - (i // n_rows) * n_rows))
        return rows^
    if n_rows <= 0:
        raise Error("uniformInt: 'end' must be greater than 'start' (rng_impl.cuh:93)")
    var rng_seed = UInt64(Int(host_seed_tree(seed, tree_id)))
    var diff = Int32(n_rows).cast[DType.uint32]() - Int32(0).cast[DType.uint32]()
    for i in range(n_sampled):
        # `uniform_int_kernel` (`core/philox.mojo:203-232`): the stride is
        # 110592, so every index at or below it owns a fresh generator on
        # subsequence `i`; an index past the stride continues its thread's.
        var sub = UInt64(i % RF_RNG_STRIDE)
        comptime if RF_ORACLE_HOST_SABOTAGE:
            # THE NEGATIVE CONTROL: the next subsequence, so every drawn row
            # moves.
            sub = sub + UInt64(1)
        var gen = host_philox_init(rng_seed, sub, UInt64(0))
        var draws = i // RF_RNG_STRIDE
        for _ in range(draws):
            _ = host_philox_uniform_int(gen, Int32(0), diff)
        rows.append(host_philox_uniform_int(gen, Int32(0), diff))
    return rows^


def _node_best_split(
    item: HostWorkItem,
    row_ids: List[Int32],
    q: RfHostQuantiles,
    labels_i: List[Int32],
    labels_f: List[Float32],
    classification: Bool,
    n_classes: Int,
    n_rows: Int,
    n_cols: Int,
    max_n_bins: Int,
    p: RfHostParams,
    criterion: Int,
    label_scale: Float32,
    tree_id: Int,
    k: Int,
    sample_offset: Int,
) raises -> HostSplit:
    """One node of one sampling round: `phase_setup_kernel`'s column sample,
    then per sampled column `build_histograms_kernel`, `pdf_to_cdf`,
    `Gain` per thread, the pinned reduction and `_publish_to_global`, then
    DEVIATION 2502's purity store. Returns the node's slot."""
    var slot = HostSplit.empty()
    var pure_flag = Int32(0)
    var node_seed = host_seed_tree_node(p.seed, tree_id, item.idx)
    var bijection = HostFeistel(n_cols, node_seed)
    for c in range(k):
        # `sampled_column_at` (`builder_kernels.mojo:315-335`).
        var col = bijection.at(sample_offset + c)
        var nb = Int(q.n_bins[col])
        var col_q = col * max_n_bins
        var planes = n_classes if classification else 1
        var counts = List[UInt32](length=nb * planes, fill=UInt32(0))
        var label_sums = List[Int32](length=nb, fill=Int32(0))
        # `_histogram_inner_loop_binned` (`builder_kernels_impl.mojo:2099-2146`)
        # with `IncrementHistogram` (`bins.mojo:341-358`, `:521-544`).
        for j in range(item.begin, item.begin + item.count):
            var row = Int(row_ids[j])
            var b = Int(q.bins[col * n_rows + row])
            if classification:
                var off = Int(labels_i[row]) * nb + b
                counts[off] = counts[off] + UInt32(1)
            else:
                label_sums[b] = label_sums[b] + Int32(labels_f[row] * label_scale)
                counts[b] = counts[b] + UInt32(1)
        # `pdf_to_cdf` per class plane (`core/block_scan.mojo:288-350`).
        var global_sample_count = Int64(0)
        var max_class_count = Int64(0)
        for pl in range(planes):
            for b in range(1, nb):
                counts[pl * nb + b] = counts[pl * nb + b] + counts[pl * nb + b - 1]
            var class_count = Int64(Int(counts[pl * nb + nb - 1]))
            if class_count > max_class_count:
                max_class_count = class_count
            global_sample_count += class_count
        if not classification:
            for b in range(1, nb):
                label_sums[b] = label_sums[b] + label_sums[b - 1]
        # `Gain` (`objectives.mojo:777-812`, `:1316-1352`): thread `t` strides
        # the bins from `t` by the block width.
        var threads = List[HostSplit](capacity=RF_TPB)
        for t in range(RF_TPB):
            var sp = HostSplit.empty()
            var i = t
            while i < nb:
                var n_left: Int64
                if classification:
                    # `count_left` (`split.mojo:205-240`).
                    var acc = UInt64(Int(counts[i]))
                    for jj in range(1, n_classes):
                        acc += UInt64(Int(counts[jj * nb + i]))
                    n_left = Int64(Int(acc))
                else:
                    n_left = Int64(Int(counts[i]))
                var n_right = global_sample_count - n_left
                var msl = Int64(p.min_samples_leaf)
                if n_left >= msl and n_right >= msl:
                    var gain: Float32
                    if classification:
                        if criterion == RF_ENTROPY:
                            gain = host_entropy_gain(counts, i, nb, n_classes)
                        else:
                            gain = host_gini_gain(counts, i, nb, n_classes)
                    elif criterion == RF_POISSON:
                        gain = host_poisson_gain(counts, label_sums, i, nb, label_scale)
                    elif criterion == RF_GAMMA:
                        gain = host_gamma_gain(counts, label_sums, i, nb, label_scale)
                    elif criterion == RF_INVERSE_GAUSSIAN:
                        gain = host_inverse_gaussian_gain(counts, label_sums, i, nb, label_scale)
                    else:
                        gain = host_mse_gain(counts, label_sums, i, nb, label_scale)
                    if gain > p.min_impurity_decrease:
                        # `update_bin` (`split.mojo:498-519`).
                        _ = sp.update(HostSplit(
                            Int32(0), q.values[col_q + i], Int32(col), gain,
                            n_left, Int64(0), Int32(i), Int32(i),
                        ))
                i += RF_TPB
            threads.append(sp)
        var node_pure = (
            classification
            and n_classes > 1
            and global_sample_count > 0
            and max_class_count == global_sample_count
            and p.min_impurity_decrease >= Float32(0)
        )
        var winner = host_pinned_reduce(threads, RF_TPB)
        winner.pure = Int32(1) if node_pure else Int32(0)
        if winner.is_valid():
            # `select_split_range_midpoint` (`split.mojo:404-429`).
            if winner.has_valid_split_range() and Int(winner.split_end) < nb:
                var mid = winner.split_start + (winner.split_end - winner.split_start + Int32(1)) // Int32(2)
                winner.quesval = q.values[col_q + Int(mid)]
                winner.split_start = mid
                winner.split_end = mid
            if slot.update(winner):
                slot.pure = winner.pure
        # The store at `block_idx.y == 0` (`builder_kernels_impl.mojo:2917-2922`);
        # every column of a node computes the same flag.
        pure_flag = Int32(1) if node_pure else Int32(0)
    slot.pure = pure_flag
    return slot


def _is_expandable(count: Int, depth: Int, leaf_counter: Int, p: RfHostParams) -> Bool:
    """`NodeQueue._is_expandable`, `builder.mojo:262-284`."""
    if depth >= p.max_depth:
        return False
    if count < p.min_samples_split:
        return False
    if p.max_leaves != -1 and leaf_counter >= p.max_leaves:
        return False
    return True


def rf_host_fit(
    var x: List[Float32],
    labels_i: List[Int32],
    labels_f: List[Float32],
    n_rows: Int,
    n_cols: Int,
    n_unique_labels: Int,
    classification: Bool,
    params: RfHostParams,
    label_scale: Float32,
    tree_start: Int = 0,
    weights: List[Float32] = List[Float32](),
) raises -> RfHostForest:
    """`fit_forest`, `ensemble/randomforest.mojo:2299-2849`, on the host.

    `x` is COLUMN-major float32 (the GPU binding's stage), `labels_i` the
    int32 class codes (classification) and `labels_f` the float32 targets
    (regression); the other is empty. `n_unique_labels` is `n_classes`, or
    1 for regression. `label_scale` is the regression plane's fixed-point
    scale; the classifier ignores it."""
    var p = params
    if tree_start < 0 or tree_start + p.n_trees > 2147483647:
        raise Error("invalid global tree range")
    if n_rows <= 0:
        raise Error("Invalid n_rows " + String(n_rows))
    if n_cols <= 0:
        raise Error("Invalid n_cols " + String(n_cols))
    if len(x) != n_rows * n_cols:
        raise Error("rf host: x must hold n_rows * n_cols values")
    if classification:
        if len(labels_i) != n_rows:
            raise Error("rf host: y must hold n_rows class codes")
        for r in range(n_rows):
            if Int(labels_i[r]) < 0 or Int(labels_i[r]) >= n_unique_labels:
                raise Error("rf host: class codes must lie in [0, n_classes)")
    elif len(labels_f) != n_rows:
        raise Error("rf host: y must hold n_rows targets")
    rf_host_check_params(p, classification)
    # No missing-value arm: a NaN bins left but partitions right, and at
    # unlimited depth the fit never returned (`has_nan_f32_threaded`).
    if has_nan_f32_threaded(
        x.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), len(x)
    ):
        raise Error(RF_NAN_REFUSAL)
    if len(weights) > 0 and not p.bootstrap:
        raise Error(
            "rf host: class weights without bootstrap reach the weighted"
            " objective (WeightedClassificationBin), which has no host"
            " restatement (ensemble/host/rf_oracle.mojo); a CPU-only install"
            " refuses it by name rather than fitting a different forest"
        )
    # `Builder.__init__`'s criterion resolution (`builder.mojo:1224-1228`).
    var criterion = p.criterion
    if criterion == RF_CRITERION_END:
        criterion = RF_GINI if classification else RF_MSE
    # `:2420-2426` -- the n_bins clamp.
    if p.max_n_bins > n_rows:
        print(
            "WARN: The number of bins, `n_bins` is greater than the number"
            " of samples used for training. Changing `n_bins` to number of"
            " training samples."
        )
        p.max_n_bins = n_rows
    # `:2429-2434`
    if not p.bootstrap and p.max_samples != Float32(1.0):
        print(
            "WARN: If bootstrap sampling is disabled, max_samples value is"
            " ignored and whole dataset is used for building each tree"
        )
        p.max_samples = Float32(1.0)
    var n_sampled = n_sampled_rows_host(p.bootstrap, p.max_samples, n_rows)
    if n_sampled <= 0:
        raise Error(
            "max_samples " + String(p.max_samples) + " x n_rows " + String(n_rows)
            + " rounds to " + String(n_sampled) + " sampled rows; a tree needs at least one"
        )
    # `ftz_features_kernel` (`:1260-1275`), in place, once.
    var n_cells = n_rows * n_cols
    var flush_tasks = host_predict_task_count(n_cells)
    if n_cells < (1 << 19):
        flush_tasks = 1
    var flush_chunk = host_predict_chunk(n_cells, flush_tasks)
    def _flush_cells(task: Int) {mut x, imm n_cells, imm flush_chunk}:
        var lo = task * flush_chunk
        var hi = min(lo + flush_chunk, n_cells)
        for i in range(lo, hi):
            x[i] = ftz(x[i])
    if flush_tasks == 1:
        _flush_cells(0)
    else:
        sync_parallelize(_flush_cells, flush_tasks)
    var q = host_compute_quantiles(x, n_rows, n_cols, p.max_n_bins, p.seed)
    # `fit_forest` calls `prepare_weights` before the first tree
    # (`randomforest.mojo:2567-2568`).
    var weight_cdf = List[Float64]()
    if len(weights) > 0:
        weight_cdf = host_weight_cdf(weights, n_rows)

    # `Builder.__init__` (`builder.mojo:1230-1242`): `n_sampled_cols_for`.
    var original_cols = Int(Float32(n_cols) * p.max_features)
    if original_cols < 1:
        original_cols = 1
    if original_cols > n_cols:
        raise Error("n_sampled_cols must be in [1, n_cols]; got " + String(original_cols))
    var max_rounds = ceildiv(n_cols, original_cols)
    var n_classes = n_unique_labels if classification else 1
    if n_classes < 1:
        raise Error("n_classes should be at least 1")

    var forest = RfHostForest()
    forest.n_trees = p.n_trees
    forest.num_outputs = n_unique_labels
    forest.offsets.append(Int32(0))
    for t in range(p.n_trees):
        var tree_id = tree_start + t
        var row_ids = host_sampled_rows(p.seed, tree_id, p.bootstrap, n_rows, n_sampled, weight_cdf)

        # `NodeQueue.__init__` (`builder.mojo:197-232`).
        var t_colid = List[Int32]()
        var t_quesval = List[Float32]()
        var t_left = List[Int32]()
        var t_count = List[Int32]()
        var r_begin = List[Int]()
        var r_count = List[Int]()
        var leaf_counter = 1
        t_colid.append(Int32(0))
        t_quesval.append(Float32(0.0))
        t_left.append(Int32(-1))
        t_count.append(Int32(n_sampled))
        r_begin.append(0)
        r_count.append(n_sampled)
        var work = List[HostWorkItem]()
        var head = 0
        if _is_expandable(n_sampled, 0, leaf_counter, p):
            work.append(HostWorkItem(0, 0, 0, n_sampled))

        while len(work) - head > 0:
            # `pop` (`:239-260`).
            var items = List[HostWorkItem]()
            while head < len(work) and len(items) < p.max_batch_size:
                items.append(work[head])
                head += 1
            var n = len(items)
            var final_splits = List[HostSplit](capacity=n)
            for _ in range(n):
                final_splits.append(HostSplit.empty())
            var active = List[Int](capacity=n)
            for i in range(n):
                active.append(i)
            var sampling_round = 0
            while True:
                var k = min(original_cols, n_cols - sampling_round * original_cols)
                var sample_offset = sampling_round * original_cols
                var retry = List[Int]()
                for a in range(len(active)):
                    var orig = active[a]
                    var s = _node_best_split(
                        items[orig], row_ids, q, labels_i, labels_f,
                        classification, n_classes, n_rows, n_cols, p.max_n_bins,
                        p, criterion, label_scale, tree_id, k, sample_offset,
                    )
                    # `_read_splits` (`builder.mojo:1733-1763`): a pure node
                    # is a leaf whatever its slot holds.
                    var terminal = s.pure != Int32(0)
                    if terminal:
                        s.colid = Int32(-1)
                    final_splits[orig] = s
                    if not s.is_valid() and not terminal:
                        retry.append(orig)
                # `advance_batch`'s retry test (`:2290-2301`).
                if len(retry) > 0 and sampling_round + 1 < max_rounds:
                    active = retry^
                    sampling_round += 1
                    continue
                break

            # `enqueue_node_split` (`builder.mojo:2357-2452`): the splits go
            # back with `split_start`/`split_end` -1 and `pure` 0, the local
            # left count is taken on valid splits, and each valid node's
            # range is stably partitioned.
            for i in range(n):
                var s = final_splits[i]
                s.pure = Int32(0)
                s.local_n_left = Int64(0)
                if s.is_valid():
                    var it = items[i]
                    var left_rows = List[Int32]()
                    var right_rows = List[Int32]()
                    for j in range(it.begin, it.begin + it.count):
                        var row = row_ids[j]
                        if x[Int(s.colid) * n_rows + Int(row)] <= s.quesval:
                            left_rows.append(row)
                        else:
                            right_rows.append(row)
                    s.local_n_left = Int64(len(left_rows))
                    var w = it.begin
                    for li in range(len(left_rows)):
                        row_ids[w] = left_rows[li]
                        w += 1
                    for ri in range(len(right_rows)):
                        row_ids[w] = right_rows[ri]
                        w += 1
                final_splits[i] = s

            # `NodeQueue.push` (`builder.mojo:286-433`).
            for i in range(n):
                var s = final_splits[i]
                var it = items[i]
                if not s.is_valid():
                    continue
                if p.max_leaves != -1 and leaf_counter >= p.max_leaves:
                    break
                var local_left_count = Int(s.local_n_left)
                var parent_begin = r_begin[it.idx]
                var parent_count = r_count[it.idx]
                var left_child_id = len(t_colid)
                t_colid[it.idx] = s.colid
                t_quesval[it.idx] = s.quesval
                t_left[it.idx] = Int32(left_child_id)
                t_count[it.idx] = Int32(parent_count)
                leaf_counter += 1
                var left_count = Int(Int32(Int(s.global_n_left)))
                t_colid.append(Int32(0))
                t_quesval.append(Float32(0.0))
                t_left.append(Int32(-1))
                t_count.append(Int32(left_count))
                r_begin.append(parent_begin)
                r_count.append(local_left_count)
                if _is_expandable(left_count, it.depth + 1, leaf_counter, p):
                    work.append(HostWorkItem(len(t_colid) - 1, it.depth + 1, parent_begin, local_left_count))
                var right_count = Int(Int32(Int(t_count[it.idx]) - Int(s.global_n_left)))
                t_colid.append(Int32(0))
                t_quesval.append(Float32(0.0))
                t_left.append(Int32(-1))
                t_count.append(Int32(right_count))
                r_begin.append(parent_begin + local_left_count)
                r_count.append(parent_count - local_left_count)
                if _is_expandable(right_count, it.depth + 1, leaf_counter, p):
                    work.append(HostWorkItem(
                        len(t_colid) - 1, it.depth + 1,
                        parent_begin + local_left_count, parent_count - local_left_count,
                    ))

        # `set_leaf_predictions` (`builder.mojo:2485-2666`) through
        # `leaf_kernel` (`builder_kernels_impl.mojo:1837-1952`); internal
        # nodes keep the zero fill.
        var n_nodes = len(t_colid)
        var n_out = n_unique_labels
        var vleaf = List[Float32](length=n_nodes * n_out, fill=Float32(0.0))
        for node in range(n_nodes):
            if t_left[node] != Int32(-1):
                continue
            var begin = r_begin[node]
            var count = r_count[node]
            if classification:
                var hist = List[UInt32](length=n_out, fill=UInt32(0))
                for j in range(begin, begin + count):
                    var lab = Int(labels_i[Int(row_ids[j])])
                    hist[lab] = hist[lab] + UInt32(1)
                # `SetLeafVector` (`objectives.mojo:814-847`).
                var total = Int64(0)
                for c in range(n_out):
                    total = total + Int64(Int(hist[c]))
                if total <= 0:
                    continue
                for c in range(n_out):
                    vleaf[node * n_out + c] = ftz(
                        Int64(Int(hist[c])).cast[DType.float32]() / total.cast[DType.float32]()
                    )
            else:
                var label_sum = Int32(0)
                var cnt = UInt32(0)
                for j in range(begin, begin + count):
                    label_sum = label_sum + Int32(labels_f[Int(row_ids[j])] * label_scale)
                    cnt = cnt + UInt32(1)
                # `SetLeafVector` (`objectives.mojo:1354-1376`).
                var weight = Int64(Int(cnt))
                if weight > 0:
                    vleaf[node * n_out] = ftz(
                        _dequantize(label_sum, label_scale) / weight.cast[DType.float32]()
                    )

        for node in range(n_nodes):
            forest.colid.append(t_colid[node])
            forest.quesval.append(t_quesval[node])
            forest.left_child.append(t_left[node])
        for v in range(n_nodes * n_out):
            forest.leaves.append(vleaf[v])
        forest.offsets.append(Int32(len(forest.colid)))
    return forest^
