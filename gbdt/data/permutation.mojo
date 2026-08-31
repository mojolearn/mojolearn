# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""`TDataPermutation`: the learn permutations, and the CTR estimation order.

PORT OF `catboost/cuda/data/permutation.{h,cpp}` at CatBoost `54a8143a`,
together with the two things it stands on: `NCatboostCuda::Shuffle`
(`cuda/data/data_utils.h:21-47`) and the generator that drives it,
`TRandom` (`libs/helpers/cpu_random.h:6-99`) over `TMersenne<ui64>`
(`util/random/mersenne64.{h,cpp}`, `util/random/common_ops.h`).

## Why a GBDT port needs a random permutation at all

CatBoost's `Borders` CTR is an ORDERED TARGET STATISTIC: a row's feature
value is built from the rows that came BEFORE it, so the value depends
entirely on what "before" means. Their answer is
`ds.GetCtrsEstimationPermutation()` -- a fresh random order per permutation
dataset -- and the CTR columns are recomputed once per permutation
(`gpu_data/doc_parallel_dataset_builder.cpp:251-262`, `permutation_count`
defaulting to 4, `boosting_options.cpp:14`). The permutation-INDEPENDENT
CTRs (FeatureFreq) are written once, from the IDENTITY order, before that
loop ever runs (`:204-206`, the `MakeSequence(ctrEstimationOrder)` at
`:206` and the `writeCtrs(..., permutationIndependent)` at `:229`).

Running the ordered statistic in ROW order is not a slower CatBoost, it is
a different and worse estimator: on data sorted by target every row's
statistic reads its own neighbourhood. That is why this file exists rather
than an `iota`.

## Their permutation 0 IS the identity, and that is not a shortcut

`FillOrder` (`permutation.cpp:7-18`) returns `std::iota` when
`Index == IdentityPermutationId()`, which is `0` (`permutation.h:81-83`).
It is safe on their side because the LEARN POOL WAS ALREADY SHUFFLED at
load: `ShuffleLearnDataIfNeeded` shuffles whenever the data has any
categorical feature and `has_time` is false
(`private/libs/algo/preprocess.cpp:161-181`, `:183-199`). That stage is
CPU-side data preparation and is upstream of everything in `catboost/cuda`,
so this port does not have it, and a caller here can hand us rows in any
order at all -- including sorted by target.

`train()` therefore takes its CTR estimation order from a NON-IDENTITY
permutation id; see the deviation block below and `PORTING.md` 55.

## The seed, and why it is worth transcribing exactly

    ui64 GetSeed() const {
        return 1664525 * GetPermutationId() + 1013904223 + BlockSize;
    }

(`permutation.h:93-95`.) Then `Shuffle` builds `TRandom rng(seed)`, calls
`rng.Advance(10)` -- ten discarded draws -- and runs the modern
Fisher-Yates from `util/random/shuffle.h:24-32`:

    for (size_t i = 1; i < sz; ++i)
        DoSwap(*(begin + i), *(begin + gen.Uniform(i + 1)));

with `Uniform(t)` the REJECTION-SAMPLED `NPrivate::GenUniform`
(`common_ops.h:48-60`), not a plain modulo. There is no oracle for this on
this box -- their GPU learner does not run on Apple silicon and their CPU
learner never exposes the order -- so the only defence is that every line
below is theirs, and that `original/ctr_permutation_check.mojo` gates
`TMersenne64` against the published MT19937-64 reference stream.

`blockSize` is 1 everywhere this port reaches: the CTR estimation
permutations come from `GetPermutation(DataProvider, permutationId)`
(`doc_parallel_dataset_builder.cpp:48`), whose `blockSize` parameter
defaults to 1 (`permutation.h:98-104`). The block arm of `Shuffle`
(`data_utils.h:31-45`) is ported anyway because it is nine lines of their
file and a hole in a ported function is how a reader learns to distrust the
whole file.

`FillGroupOrder` and `GenerateQueryDocsOrder` are NOT ported: both require
`ObjectsGrouping`, and no groupwise loss is ported, so `FillOrder`'s
group branch (`permutation.cpp:9-11`) is unreachable here.
"""


comptime MT_NN = 312
"""`TMersenne64::NN` (`mersenne64.h:9`)."""

comptime MT_MM = 156
"""`#define MM 156` (`mersenne64.cpp:6`)."""

comptime MT_MATRIX_A = UInt64(0xB5026F5AA96619E9)
"""`#define MATRIX_A` (`mersenne64.cpp:7`)."""

comptime MT_UM = UInt64(0xFFFFFFFF80000000)
"""`#define UM` (`mersenne64.cpp:8`), the most significant 33 bits."""

comptime MT_LM = UInt64(0x7FFFFFFF)
"""`#define LM` (`mersenne64.cpp:9`), the least significant 31 bits."""


struct TMersenne64(Movable):
    """`NPrivate::TMersenne64` (`util/random/mersenne64.{h,cpp}`).

    Plain MT19937-64. `InitByArray` and the `IInputStream` constructor are
    not ported: `TRandom` only ever calls the scalar-seed one
    (`cpu_random.h:8-11`).
    """

    var mt: List[UInt64]
    var mti: Int

    def __init__(out self, seed: UInt64):
        """`TMersenne64(ui64 s)` (`mersenne64.h:12-16`): `mti` is set to
        `NN + 1` by the initializer list and then overwritten by
        `InitGenRand`, which leaves it at `NN`. So the first `GenRand`
        twists immediately and never sees the `NN + 1` fallback seeding at
        `mersenne64.cpp:68-70`."""
        self.mt = List[UInt64]()
        for _ in range(MT_NN):
            self.mt.append(UInt64(0))
        self.mti = MT_NN + 1
        self.init_gen_rand(seed)

    def init_gen_rand(mut self, seed: UInt64):
        """`InitGenRand` (`mersenne64.cpp:13-19`)."""
        self.mt[0] = seed
        var i = 1
        while i < MT_NN:
            var prev = self.mt[i - 1]
            self.mt[i] = UInt64(6364136223846793005) * (
                prev ^ (prev >> 62)
            ) + UInt64(i)
            i += 1
        self.mti = i

    def init_next(mut self):
        """`InitNext` (`mersenne64.cpp:60-86`), the twist.

        `mag01[x & 1]` is written as a branch rather than a two-element
        table; it is the same select and the table is a C idiom, not part
        of the algorithm.
        """
        if self.mti == MT_NN + 1:
            self.init_gen_rand(UInt64(5489))

        var i = 0
        while i < MT_NN - MT_MM:
            var x = (self.mt[i] & MT_UM) | (self.mt[i + 1] & MT_LM)
            var m = MT_MATRIX_A if (x & UInt64(1)) != UInt64(0) else UInt64(0)
            self.mt[i] = self.mt[i + MT_MM] ^ (x >> 1) ^ m
            i += 1
        while i < MT_NN - 1:
            var x = (self.mt[i] & MT_UM) | (self.mt[i + 1] & MT_LM)
            var m = MT_MATRIX_A if (x & UInt64(1)) != UInt64(0) else UInt64(0)
            self.mt[i] = self.mt[i + (MT_MM - MT_NN)] ^ (x >> 1) ^ m
            i += 1

        var x = (self.mt[MT_NN - 1] & MT_UM) | (self.mt[0] & MT_LM)
        var m = MT_MATRIX_A if (x & UInt64(1)) != UInt64(0) else UInt64(0)
        self.mt[MT_NN - 1] = self.mt[MT_MM - 1] ^ (x >> 1) ^ m

        self.mti = 0

    def gen_rand(mut self) -> UInt64:
        """`GenRand` (`mersenne64.h:26-39`), tempering included."""
        if self.mti >= MT_NN:
            self.init_next()

        var x = self.mt[self.mti]
        self.mti += 1

        x ^= (x >> 29) & UInt64(0x5555555555555555)
        x ^= (x << 17) & UInt64(0x71D67FFFEDA60000)
        x ^= (x << 37) & UInt64(0xFFF7EEE000000000)
        x ^= x >> 43

        return x


struct TRandom(Movable):
    """`TRandom` (`catboost/libs/helpers/cpu_random.h:6-99`), the three
    members `Shuffle` uses.

    The gaussian/gamma/beta/poisson block is not ported: none of it is
    reachable from a permutation, and `random_gen.mojo` already carries the
    device-side draws the bootstrap needs.
    """

    var rng: TMersenne64

    def __init__(out self, seed: UInt64):
        self.rng = TMersenne64(seed)

    def next_uniform_l(mut self) -> UInt64:
        """`NextUniformL()` (`cpu_random.h:23-25`) -> `GenRand64()`, which
        for a `ui64` engine is `ToRand64(engine, engine.GenRand())` and
        that overload returns `x` unchanged (`common_ops.h:40-43`)."""
        return self.rng.gen_rand()

    def advance(mut self, n: Int):
        """`Advance(ui32 n)` (`cpu_random.h:17-21`): n discarded draws."""
        for _ in range(n):
            _ = self.next_uniform_l()

    def uniform(mut self, size: UInt64) raises -> UInt64:
        """`Uniform(ui64 size)`, their rejection-sampled `GenUniform`.

        `cpu_random.h:31-33` -> `TCommonRNG::Uniform`
        (`common_ops.h:84-86`) -> `NPrivate::GenUniform`
        (`common_ops.h:48-60`), transcribed:

            const T randmax = gen.RandMax() - gen.RandMax() % max;
            while ((rand = gen.GenRand()) >= randmax) { }
            return rand % max;

        **REJECTION SAMPLING, not a plain modulo.** `RandMax()` is
        `TResult(-1)` (`common_ops.h:72-77`), so for a `ui64` engine it is
        `0xFFFFFFFFFFFFFFFF` -- one less than 2^64, which is what makes the
        remainder term non-trivial. Substituting `GenRand() % max` would
        agree with theirs on almost every draw and disagree on the tail,
        which is the shape of divergence nothing here could ever see.
        """
        if size == UInt64(0):
            # `Y_ABORT_UNLESS(max > 0, "Invalid random number range [0, 0)")`
            raise Error("Invalid random number range [0, 0)")
        var rand_max = UInt64(0xFFFFFFFFFFFFFFFF)
        var bound = rand_max - (rand_max % size)
        while True:
            var r = self.rng.gen_rand()
            if r < bound:
                return r % size


def shuffle(
    seed: UInt64, block_size: Int, sample_count: Int
) raises -> List[UInt32]:
    """`NCatboostCuda::Shuffle` (`cuda/data/data_utils.h:21-47`).

        TRandom rng(seed);
        rng.Advance(10);
        order.yresize(sampleCount);
        std::iota(order.begin(), order.end(), 0);
        if (blockSize == 1) {
            ::Shuffle(order.begin(), order.begin() + sampleCount, rng);
        } else {
            ... shuffle whole BLOCKS, then lay them out in order ...
        }

    The `blockSize == 1` arm's inner loop is `util/random/shuffle.h:24-32`:
    `for (i = 1; i < sz; ++i) DoSwap(order[i], order[gen.Uniform(i + 1)])`.
    """
    var rng = TRandom(seed)
    rng.advance(10)

    var order = List[UInt32]()
    for i in range(sample_count):
        order.append(UInt32(i))

    if block_size == 1:
        # `::Shuffle(begin, begin + sampleCount, rng)` (`shuffle.h:24-32`)
        for i in range(1, sample_count):
            var j = Int(rng.uniform(UInt64(i + 1)))
            var t = order[i]
            order[i] = order[j]
            order[j] = t
    else:
        # `NHelpers::CeilDivide(order.size(), blockSize)` (`data_utils.h:32`)
        var blocks_count = (sample_count + block_size - 1) // block_size
        var blocks = List[UInt32]()
        for i in range(blocks_count):
            blocks.append(UInt32(i))
        for i in range(1, blocks_count):
            var j = Int(rng.uniform(UInt64(i + 1)))
            var t = blocks[i]
            blocks[i] = blocks[j]
            blocks[j] = t

        var cursor = 0
        for i in range(blocks_count):
            var block_start = Int(blocks[i]) * block_size
            var block_end = block_start + block_size
            if block_end > sample_count:
                block_end = sample_count
            for j in range(block_start, block_end):
                order[cursor] = UInt32(j)
                cursor += 1
    return order^


comptime IDENTITY_PERMUTATION_ID = 0
"""`TDataPermutation::IdentityPermutationId()` (`permutation.h:81-83`)."""


struct TDataPermutation(Copyable, Movable):
    """`NCatboostCuda::TDataPermutation` (`permutation.h:11-96`).

    `DataProvider` becomes a plain document count: the only two things the
    class asks it for are `GetObjectCount()` and the grouping, and the
    grouping branch is unported (see the file header).
    """

    var doc_count: Int
    var index: Int
    var block_size: Int

    def __init__(out self, doc_count: Int, index: Int, block_size: Int):
        """`TDataPermutation(dataProvider, index, blockSize)`
        (`permutation.h:18-25`)."""
        self.doc_count = doc_count
        self.index = index
        self.block_size = block_size

    def get_seed(self) -> UInt64:
        """`GetSeed()` (`permutation.h:93-95`):
        `1664525 * GetPermutationId() + 1013904223 + BlockSize`."""
        return UInt64(1664525) * UInt64(self.index) + UInt64(
            1013904223
        ) + UInt64(self.block_size)

    def is_identity(self) -> Bool:
        """`IsIdentity()` (`permutation.h:77-79`)."""
        return self.index == IDENTITY_PERMUTATION_ID

    def get_permutation_id(self) -> Int:
        """`GetPermutationId()` (`permutation.h:85-87`)."""
        return self.index

    def fill_order(self) raises -> List[UInt32]:
        """`FillOrder(TVector<ui32>& order)` (`permutation.cpp:7-18`).

        `order[i]` is the ORIGINAL row that sits at position `i`. That is
        the direction `WriteOrder` hands to `ctrEstimationOrder`
        (`permutation.h:63-68`), which is the direction
        `TCtrBinBuilder::SetIndices` consumes, which is the direction the
        segmented scan's `Index()` reads back out. Inverting it here would
        be invisible on any check whose permutation is an involution, so
        the direction is written down rather than left to be inferred.
        """
        if self.index != IDENTITY_PERMUTATION_ID:
            return shuffle(self.get_seed(), self.block_size, self.doc_count)
        var order = List[UInt32]()
        for i in range(self.doc_count):
            order.append(UInt32(i))
        return order^

    def fill_inverse_permutation(self) raises -> List[UInt32]:
        """`FillInversePermutation` (`permutation.cpp:30-37`)."""
        var order = self.fill_order()
        var inverse = List[UInt32]()
        for _ in range(len(order)):
            inverse.append(UInt32(0))
        for i in range(len(order)):
            inverse[Int(order[i])] = UInt32(i)
        return inverse^


def get_permutation(
    doc_count: Int, permutation_id: Int, block_size: Int = 1
) -> TDataPermutation:
    """`GetPermutation(dataProvider, permutationId, blockSize = 1)`
    (`permutation.h:98-104`). The default `blockSize` of 1 is theirs and is
    what `doc_parallel_dataset_builder.cpp:48` takes."""
    return TDataPermutation(doc_count, permutation_id, block_size)


def get_identity_permutation(doc_count: Int) -> TDataPermutation:
    """`GetIdentityPermutation(dataProvider)` (`permutation.h:106-110`)."""
    return TDataPermutation(doc_count, IDENTITY_PERMUTATION_ID, 1)


comptime DEFAULT_PERMUTATION_COUNT = 4
"""`PermutationCount("permutation_count", 4)` (`boosting_options.cpp:14`).

It is NOT reduced to 1 on this path: `UpdateGpuSpecificDefaults` collapses
it only when the feature manager has no permutation features AND boosting is
Plain (`cuda/train_lib/train.cpp:102-107`), and a `Borders` CTR is exactly a
permutation feature. `DataProcessingOptions->HasTimeFlag` also forces 1
(`catboost_options.cpp:1043-1045`) and this port has no `has_time`.
"""


def ctrs_estimation_permutation(
    doc_count: Int, permutation_id: Int
) -> TDataPermutation:
    """`ds.GetCtrsEstimationPermutation()` for permutation `permutation_id`
    (`gpu_data/dataset_base.h:250-252`, built at
    `doc_parallel_dataset_builder.cpp:45-54` from
    `GetPermutation(DataProvider, permutationId)`).

    Their loop runs this for every `permutationId` in
    `[0, permutation_count)` and writes a SEPARATE set of CTR columns per
    permutation into that permutation's own compressed dataset
    (`:251-262`). See `PORTING.md` 55 for which one this port keeps.
    """
    return get_permutation(doc_count, permutation_id, 1)
