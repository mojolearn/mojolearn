# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`TFeatureTensor` and `TBatchFeatureTensorBuilder`: what a TREE CTR's
source actually is, and the loop that enumerates one per categorical
feature.

MIRRORS `catboost/cuda/data/feature.h:89-188` (`TFeatureTensor`, and the
`THash` specialisation under it) and
`catboost/cuda/methods/batch_feature_tensor_builder.{h,cpp}` at CatBoost
`54a8143a`. Transliterated. Do not improve.

## What a tensor is, and why it cannot be a preprocessing pass

A CTR turns a categorical column into NUMERIC columns. A SIMPLE ctr's
source is one categorical feature. A TREE ctr's source is a TENSOR: the
binary splits ALREADY IN THE CURRENT TREE, crossed with a categorical
feature. The candidate set therefore grows level by level and the values
have to be built INSIDE the searcher loop, which is why tree CTRs exist
only in `TFeatureParallelObliviousTreeSearcher`
(`oblivious_tree_structure_searcher.cpp:106-209`) and are absent from both
of the stripe-mapped searchers.

Two predicates in `binarizations_manager.h` gate the whole block:

    IsTreeCtrsEnabled()          catFeatures present && MaxTensorComplexity > 1   (:65-68)
    UseAsBaseTensorForTreeCtr()  tensor.GetComplexity() < MaxTensorComplexity     (:70-72)

and `MaxTensorComplexity` defaults to 4 (`cat_feature_options.cpp:231`).
**This port pins `max_ctr_complexity` at 1 and
`TCatFeatureParams.check()` refuses anything larger, so the first
predicate is false and NOTHING BELOW RUNS DURING A FIT.** That pin is not
changed here; see the file's tail comment for what would have to move.

## The four things a tensor is, in their order

1. **A canonical VALUE.** Every mutator ends in `SortUniqueSplits()` /
   `SortUniqueCatFeatures()` (`feature.h:95-131`), so the stored form is
   the sorted, deduplicated one and the INSERTION ORDER IS ERASED. Two
   tensors naming the same splits and the same categorical features are
   `==` and hash equal however they were built. Their dataset cache
   depends on exactly that: `TTreeCtrDataSet` is keyed by tensor, so an
   order-dependent identity would silently recompute (a miss that should
   have hit) or silently alias (a hit that should have missed).

2. **An ORDER.** `operator<` is `std::tie(Splits, CatFeatures)`
   (`feature.h:165-167`), lexicographic on the canonical vectors, and
   `IsSubset` is two `std::includes` over the same sorted ranges
   (`helpers/set.h:7-10`).

3. **A HASH**, `MultiHash(TVecHash<TBinarySplit>()(Splits),
   VecCityHash(CatFeatures))` (`feature.h:161-163`). Two different mixers
   on the two halves, both transcribed below.

4. **A COMPLEXITY**, and it is not what it looks like:
   `CatFeatures.size() + min(Splits.size(), 1)` (`feature.h:181-183`).
   **EVERY SPLIT IN THE TENSOR TOGETHER COUNTS ONE.** A tensor holding a
   depth-6 tree's six splits plus one categorical feature has complexity
   2, not 7, and is still usable as a base at the default
   `MaxTensorComplexity` of 4. `Size()` (`:157-159`) is the plain sum and
   is a DIFFERENT number; the two are one line apart in their header and
   nothing but the name distinguishes them at a call site.

## The hash chain, spelled out because every link has a trap in it

    THash<ui32>(x)            = (size_t)x                 str_stl.h:38-47 (integral)
    THash<EBinSplitType>(x)   = NumericHash(x)            str_stl.h:49-54 (scalar, non-integral)
                              = IntHashImpl((ui32)x)      numeric.h:39-48, 73-88
    CombineHashes(l, r)       = IntHashImpl((ui64)l) ^ r  numeric.h:63-68, 90-93
    MultiHash(a, b, c)        = CombineHashes(CombineHashes(THash(c), THash(b)), THash(a))
                                                          multi.h:6-14

so `TBinarySplit::GetHash()` (`feature.h:67-69`) folds RIGHT TO LEFT --
the split type is hashed first and the feature id is XORed in last -- and
the two `ui32` members enter the fold as their bare values while the enum
enters through a mixer. Getting the fold order backwards produces a
perfectly good-looking hash that agrees with theirs on nothing.

`TVecHash<TBinarySplit>` (`libs/model/hash.h:16-25`) has two quirks that
are not typos:

* it accumulates in `ui32` (`res = 984121 * res + a[i].GetHash()`), so
  each split's 64-bit hash is TRUNCATED to its low 32 bits;
* it returns **`int`**, and `THash<int>` is the plain `(size_t)` cast, so
  a result at or above 2^31 enters `MultiHash` SIGN-EXTENDED as
  `0xffffffff________`. Half of the fixtures in
  `checks/feature_tensor_check.mojo` land there. This is the same
  defect `gbdt/models/hash.mojo:cat_hash_chain_element` documents for
  `ctr_provider.h:107`, at a second site, and it is written here as the
  same explicit branch for the same reason: a chained SIMD cast got the
  first one wrong.

`VecCityHash` (`libs/helpers/hash.h:6-9`) hashes the RAW BYTES of the
`ui32` vector -- `CityHash64(data(), 4 * size())` -- so it is
byte-order-dependent by construction and this port assembles the bytes
little-endian, exactly as `gbdt/digest/city.mojo` already documents for
every other reader of that function. An EMPTY cat vector hashes `k2`
(`city.cpp:96`), not zero.

## HAZARD, found by this lane's own adversarial fixture: `UInt64(f(x))` sign-extends

MEASURED 2026-08-21 on Mojo 1.0. Given

    def _as_u32(v: Int32) -> UInt32:
        return v.cast[DType.uint32]()

these two lines produce DIFFERENT 64-bit values for `v == -1`:

    UInt64(_as_u32(v))              -> 0xffffffffffffffff     WRONG
    var u = _as_u32(v); UInt64(u)   -> 0x00000000ffffffff     right

The same-width `int32 -> uint32` cast is elided at the call site, so the
`UInt64` constructor binds against the ORIGINAL `Int32` and sign-extends.
It reproduces only when the callee's body IS that cast: `UInt64(f())` for
an `f` returning a `UInt32` LITERAL is correct, and so is any version that
binds the intermediate to a `var`. A function whose entire job is to change
signedness is therefore not safe to call inline.

It cost this file a wrong `TBinarySplit::GetHash` for every split with a
field at or above 2^31, and NOTHING ELSE IN THE CHECK MOVED: the canonical
form, the comparator, `IsSubset` and both scalar answers were all still
green, because they compare two values that were mangled identically. One
fixture with `0x80000001` in it is what caught it.

Two defences, both applied: `@no_inline` on `_as_u32`, and `_widen_u32`,
which masks to 32 bits so the answer is right whichever way the
constructor resolves. This is the third member of the family
`gbdt/models/hash.mojo` and `archive/reference/PORTING.md` 17 opened -- **assume Mojo's
numeric conversions are approximate until an external oracle says
otherwise.**

## `TBatchFeatureTensorBuilder`

`VisitCtrBinBuilders` (`batch_feature_tensor_builder.cpp:10-65`) is the
only place tree-CTR tensors are minted. Its shape:

    ComputeCurrentBins(baseTensorIndices, ...)      -> once, for the batch
    buildStreams = min(TensorBuilderStreams, catFeatureIds.size())
    for i in 0, buildStreams, 2*buildStreams, ...
        for j in 0 .. buildStreams:   CtrBinBuilders[j].SetIndices(base)
                                                       .AddCompressedBinsWithCurrentBinsCache(...)
        for j in 0 .. buildStreams:   tensor = baseTensor; tensor.AddCatFeature(id)
                                      visitor(tensor, CtrBinBuilders[j])

The two inner loops are deliberately NOT merged -- their comment says so
outright ("do not merge with second part. ctrBinBuilder should be async
wrt host") -- because the first submits work on `buildStreams` streams and
the second is what waits on it. Merging them serialises the batch. The
split is kept here even though this port has one queue, because it is the
structure that decides WHICH builder object serves WHICH feature, and that
mapping is what a visitor sees.

Note that `RequestStream` (`:67-77`) GROWS the pools and never shrinks
them, and that the builders are member state reused across calls: slot `j`
holds whatever the previous batch left in it until `SetIndices` resets it.

## DEVIATION 116: this file holds `TFeatureTensor`, and its splits are `Int32`

THEIRS: `TFeatureTensor` and `TBinarySplit` are both in
`catboost/cuda/data/feature.h`, and both members of `TBinarySplit` are
`ui32`.

OURS: `TBinarySplit` landed earlier in `gbdt/models/oblivious_model.mojo`
with `Int32` fields, because the model is where this port first needed it.
`TFeatureTensor` is here rather than in a new `gbdt/data/feature.mojo`
because this lane ships two files and a third would be a directory
decision, not a port decision.

WHY IT MATTERS AND WHAT IT COSTS: `Int32` is the wrong signedness for
BOTH the comparator and the hash. `std::tie(FeatureId, BinIdx, SplitType)`
orders `0x80000001` ABOVE `0x7fffffff`; read as `Int32` it orders below,
which reverses the canonical form and therefore the hash. So every read of
those fields in this file goes through `_as_u32`, and
`checks/feature_tensor_check.mojo` carries a fixture whose splits are
`0x7fffffff` and `0x80000001` for no other reason than to fail if one of
those reads is ever dropped. Moving `TBinarySplit` to `UInt32` is the real
fix and belongs to whoever next touches the model.

## DEVIATION 117: `RequestStream` returns a batch width, not a stream

THEIRS: `RequestStream` calls `GetCudaManager().RequestStream()` once per
new slot and hands each `TCtrBinBuilder` its stream id, so the `j` loop
submits `buildStreams` independent bin builds concurrently.

OURS: there are no streams. `ctx.stream()` raises on Metal (PORTING_RULES
rule 4) and this port runs one queue, so `builder_streams[j]` holds the
slot index `j` and the batch is built serially. The BATCH WIDTH is kept
and so is the two-loop structure, because `buildStreams` decides the
grouping -- how many features share a pass and which builder object each
one lands in -- and that grouping is observable to the visitor whether or
not the passes overlap in time. Deleting it would make the second and
later groups unreachable, which is the specific way this repository has
been bitten five times in two days.

COST: none in output, `buildStreams` passes of latency instead of one.

## DEVIATION 118: dense cat codes, and no `currentBins` cache

THEIRS: the builder reads `TCompressedCatFeatureDataSet` (packed `ui64`
blocks, GPU or CPU resident) and calls
`AddCompressedBinsWithCurrentBinsCache(currentBins, ...)`
(`ctr_bins_builder.h:113-125`), passing the `currentBins` computed ONCE
before the loop.

OURS: `gbdt/ctrs/ctr_bins_builder.mojo` holds dense category codes rather
than their packed blocks -- that decompression deviation is already
recorded there -- and its `add_cat_feature_bins` is their
`ProceedNewBins(uniqueValues)`, the arm that recomputes `CurrentBins` from
its own `Indices` first.

WHY THAT IS VALUE-IDENTICAL, not merely close: the loop resets the builder
to `baseTensorIndices` immediately before every add, so
`ComputeCurrentBins(Indices)` and `ComputeCurrentBins(baseTensorIndices)`
read the same array. The cache is a saved pass, not a different answer.
COST: one extra O(rows) pass per categorical feature instead of one per
batch. `_set_indices` below is their `SetIndices` (`:32-52`) written as a
free function so that this lane touches no file in `gbdt/ctrs/`.

## What still stands between this and a reachable tree CTR

Nothing in this file is called by `train()`. To make
`max_ctr_complexity > 1` reachable, in order:

1. `gbdt/options/catboost_options.mojo:987` -- `TCatFeatureParams.check()`
   raises on any value but 1. That guard is correct today and must be the
   LAST thing removed, not the first.
2. A `TBinarizedFeaturesManager` with a tensor->feature-id map
   (`InverseCtrs`, `binarizations_manager.h:80-100`), which is what makes
   `IsTreeCtr(featureId)` answerable and what a split on a tree CTR is
   named by.
3. `tree_ctrs_dataset.{h,cpp}` + `tree_ctrs.{h,cpp}` -- the per-device
   dataset cache keyed by the hash below, and the memory estimator that
   decides how many tensors fit.
4. `ctr_from_tensor_calcer.h` -- the visitor this file's `visit` trait
   stands for, which turns (tensor, bin builder) into CTR columns.
5. `TFeatureTensorTracker` (`gpu_data/oblivious_tree_bin_builder.h:32-90`)
   -- the thing that produces `baseTensorIndices` at all.
6. `TFeatureParallelObliviousTreeSearcher` -- NEXT_TWO rung 2. Tree CTRs
   live nowhere else.
"""

from gbdt.ctrs.ctr_bins_builder import TCtrBinBuilder
from gbdt.digest.city import city_hash_64
from gbdt.models.oblivious_model import TBinarySplit


# --- their integer mixers (`util/digest/numeric.h`) ----------------------


@no_inline
def _as_u32(v: Int32) -> UInt32:
    """Read one of `TBinarySplit`'s `Int32` fields as the `ui32` it is in
    their struct. DEVIATION 116; every field read in this file goes
    through here.

    `@no_inline` IS LOAD-BEARING -- see the HAZARD block in the module
    docstring. Without it this function is elided at its call sites and
    the next conversion re-reads the ORIGINAL `Int32`.
    """
    return v.cast[DType.uint32]()


def _widen_u32(v: UInt32) -> UInt64:
    """Widen a `ui32` to `ui64` with a ZERO extension, which is what
    `THash<ui32>`'s `(size_t)` does (`str_stl.h:38-47`).

    The `& 0xFFFFFFFF` is not decoration: `UInt64(f(x))` where `f` returns
    a `UInt32` produced by a same-width cast SIGN-extended in Mojo 1.0 --
    the HAZARD block in the module docstring has the measurement. The mask
    makes the answer right whichever way the compiler resolves the
    constructor, which is the property this file needs.
    """
    var w = UInt64(v)
    return w & UInt64(0xFFFFFFFF)


def int_hash_impl_u32(key_in: UInt32) -> UInt32:
    """`IntHashImpl(ui32)` (`numeric.h:39-48`), Thomas Wang's 32-bit
    mixer."""
    var key = key_in
    key += ~(key << 15)
    key ^= key >> 10
    key += key << 3
    key ^= key >> 6
    key += ~(key << 11)
    key ^= key >> 16
    return key


def int_hash_impl_u64(key_in: UInt64) -> UInt64:
    """`IntHashImpl(ui64)` (`numeric.h:50-62`)."""
    var key = key_in
    key += ~(key << 32)
    key ^= key >> 22
    key += ~(key << 13)
    key ^= key >> 8
    key += key << 3
    key ^= key >> 15
    key += ~(key << 27)
    key ^= key >> 31
    return key


def combine_hashes(l: UInt64, r: UInt64) -> UInt64:
    """`CombineHashes(T l, T r) = IntHash(l) ^ r` (`numeric.h:90-93`) at
    `T = size_t`.

    `IntHash<T>` converts to the SAME-WIDTH unsigned first
    (`TFixedWidthUnsignedInt<T>`, `numeric.h:63-68`), which is what picks
    the 64-bit mixer here. It is NOT symmetric: only the left argument is
    mixed, so swapping the two produces a different, equally plausible
    number.
    """
    return int_hash_impl_u64(l) ^ r


def sign_extend_i32(v: Int32) -> UInt64:
    """The `(size_t)` of `THash<int>` (`str_stl.h:38-47`, the integral
    `THashHelper`), written as an explicit branch.

    `TVecHash` returns `int`. Widening a negative one to `size_t` SIGN
    extends. Chained SIMD casts got this exact conversion wrong once
    already at the other site that needs it
    (`gbdt/models/hash.mojo:cat_hash_chain_element`), so it is spelled out
    rather than trusted to a cast chain.
    """
    var u = v.cast[DType.uint32]()
    var w = _widen_u32(u)
    if (u & UInt32(0x80000000)) != UInt32(0):
        return w | UInt64(0xFFFFFFFF00000000)
    return w


def binary_split_hash(split: TBinarySplit) -> UInt64:
    """`TBinarySplit::GetHash()` (`feature.h:67-69`), which is
    `MultiHash(FeatureId, BinIdx, SplitType)`.

    `MultiHash` folds from the TAIL (`multi.h:11-14`): the last argument
    is hashed alone, then each earlier one is combined in on the LEFT. So
    the split type goes through `NumericHash` (it is an `enum class`,
    scalar and non-integral, `str_stl.h:48-53`) and the two `ui32`s go in
    as their bare values.
    """
    var mixed = int_hash_impl_u32(_as_u32(split.split_type))
    var h = _widen_u32(mixed)
    h = combine_hashes(h, _widen_u32(_as_u32(split.bin_idx)))
    h = combine_hashes(h, _widen_u32(_as_u32(split.feature_id)))
    return h


def t_vec_hash_splits(splits: List[TBinarySplit]) -> Int32:
    """`TVecHash<TBinarySplit>` (`libs/model/hash.h:16-25`).

    Both surprises are theirs and both are load-bearing: the accumulator
    is `ui32`, so each split's 64-bit hash is truncated to its low half on
    the way in, and the return type is SIGNED `int`.
    """
    var res = UInt32(1988712)
    for i in range(len(splits)):
        res = UInt32(984121) * res + binary_split_hash(splits[i]).cast[
            DType.uint32
        ]()
    return res.cast[DType.int32]()


def vec_city_hash_u32(values: List[UInt32]) -> UInt64:
    """`VecCityHash` (`libs/helpers/hash.h:6-9`):
    `CityHash64((const char*)data(), sizeof(ui32) * size())`.

    The RAW BYTES of the vector, so the layout is part of the hash. Every
    target this repository ships on is little-endian and
    `gbdt/digest/city.mojo` already says so; the bytes are assembled that
    way here rather than punned, because there is no portable pun.

    An empty vector hashes `k2 == 0x9ae16a3b2f90404f` (`city.cpp:96`), and
    a tensor with no categorical features is the common case, so that
    branch is on the default path rather than an edge.
    """
    var bytes = List[UInt8]()
    for i in range(len(values)):
        var v = values[i]
        bytes.append((v & UInt32(0xFF)).cast[DType.uint8]())
        bytes.append(((v >> 8) & UInt32(0xFF)).cast[DType.uint8]())
        bytes.append(((v >> 16) & UInt32(0xFF)).cast[DType.uint8]())
        bytes.append(((v >> 24) & UInt32(0xFF)).cast[DType.uint8]())
    return city_hash_64(Span(bytes))


# --- `TBinarySplit`'s order and equality (`feature.h:50-64`) -------------


def binary_split_less(a: TBinarySplit, b: TBinarySplit) -> Bool:
    """Their `operator<`, `std::tie(FeatureId, BinIdx, SplitType)`
    (`feature.h:51-53`). UNSIGNED on all three, DEVIATION 116."""
    var af = _as_u32(a.feature_id)
    var bf = _as_u32(b.feature_id)
    if af != bf:
        return af < bf
    var ab = _as_u32(a.bin_idx)
    var bb = _as_u32(b.bin_idx)
    if ab != bb:
        return ab < bb
    return _as_u32(a.split_type) < _as_u32(b.split_type)


def binary_split_equal(a: TBinarySplit, b: TBinarySplit) -> Bool:
    """Their `operator==` (`feature.h:59-61`), all THREE members."""
    return (
        _as_u32(a.feature_id) == _as_u32(b.feature_id)
        and _as_u32(a.bin_idx) == _as_u32(b.bin_idx)
        and _as_u32(a.split_type) == _as_u32(b.split_type)
    )


# --- `TFeatureTensor` (`feature.h:89-188`) ------------------------------


struct TFeatureTensor(Copyable, Movable):
    """Their `TFeatureTensor`. Two sorted, deduplicated vectors and
    nothing else.

    Their members are private behind `GetSplits()` / `GetCatFeatures()`
    and every mutator restores the invariant before returning. Mojo has no
    access control, so the invariant is stated here instead: **`splits`
    and `cat_features` are sorted and unique at every point a caller can
    observe them.** Writing either field directly breaks the hash, the
    comparator and `is_subset` all at once, and nothing will say so.
    """

    var splits: List[TBinarySplit]
    var cat_features: List[UInt32]

    def __init__(out self):
        self.splits = List[TBinarySplit]()
        self.cat_features = List[UInt32]()

    # -- the mutators (`feature.h:95-131`) --------------------------------

    def sort_unique_splits(mut self):
        """`SortUniqueSplits()` (`feature.h:109-112`): `Sort` then their
        `Unique`, which is `std::unique` + `resize`
        (`feature.h:83-87`) and therefore removes only CONSECUTIVE
        duplicates. That is a full dedup only because the sort ran first;
        the two lines are one algorithm and neither survives alone.

        Their `Sort` is `std::sort`. An insertion sort stands in because
        the vector is at most `max_depth` long -- eight, at their default
        -- and because the comparator is a strict weak order with exact
        equality under `binary_split_equal`, so no sort of it can produce
        a different canonical form.
        """
        var n = len(self.splits)
        for i in range(1, n):
            var j = i
            while j > 0 and binary_split_less(self.splits[j], self.splits[j - 1]):
                var tmp = self.splits[j]
                self.splits[j] = self.splits[j - 1]
                self.splits[j - 1] = tmp
                j -= 1
        var out = List[TBinarySplit]()
        for i in range(n):
            if i == 0 or not binary_split_equal(self.splits[i], self.splits[i - 1]):
                out.append(self.splits[i])
        self.splits = out^

    def sort_unique_cat_features(mut self):
        """`SortUniqueCatFeatures()` (`feature.h:128-131`)."""
        var n = len(self.cat_features)
        for i in range(1, n):
            var j = i
            while j > 0 and self.cat_features[j] < self.cat_features[j - 1]:
                var tmp = self.cat_features[j]
                self.cat_features[j] = self.cat_features[j - 1]
                self.cat_features[j - 1] = tmp
                j -= 1
        var out = List[UInt32]()
        for i in range(n):
            if i == 0 or self.cat_features[i] != self.cat_features[i - 1]:
                out.append(self.cat_features[i])
        self.cat_features = out^

    def add_binary_split(mut self, bin: TBinarySplit):
        """`AddBinarySplit(const TBinarySplit&)` (`feature.h:95-99`)."""
        self.splits.append(bin)
        self.sort_unique_splits()

    def add_binary_splits(mut self, splits: List[TBinarySplit]):
        """`AddBinarySplit(const TVector<TBinarySplit>&)`
        (`feature.h:101-107`): push ALL of them, then sort-unique ONCE."""
        for i in range(len(splits)):
            self.splits.append(splits[i])
        self.sort_unique_splits()

    def add_cat_feature(mut self, feature_id: UInt32):
        """`AddCatFeature(ui32)` (`feature.h:122-126`). This is the one the
        batch builder calls, once per categorical feature, on a copy of the
        base tensor."""
        self.cat_features.append(feature_id)
        self.sort_unique_cat_features()

    def add_cat_features(mut self, feature_ids: List[UInt32]):
        """`AddCatFeature(const TVector<ui32>&)` (`feature.h:114-120`)."""
        for i in range(len(feature_ids)):
            self.cat_features.append(feature_ids[i])
        self.sort_unique_cat_features()

    def add_tensor(mut self, other: TFeatureTensor):
        """`AddTensor(const TFeatureTensor&)` (`feature.h:133-143`): append
        both vectors, then sort-unique both. The UNION, not a
        concatenation -- a feature the two tensors share appears once."""
        for i in range(len(other.splits)):
            self.splits.append(other.splits[i])
        for i in range(len(other.cat_features)):
            self.cat_features.append(other.cat_features[i])
        self.sort_unique_splits()
        self.sort_unique_cat_features()

    # -- identity ---------------------------------------------------------

    def __eq__(self, other: Self) -> Bool:
        """`operator==` (`feature.h:145-147`): both canonical vectors,
        element for element."""
        if len(self.splits) != len(other.splits):
            return False
        if len(self.cat_features) != len(other.cat_features):
            return False
        for i in range(len(self.splits)):
            if not binary_split_equal(self.splits[i], other.splits[i]):
                return False
        for i in range(len(self.cat_features)):
            if self.cat_features[i] != other.cat_features[i]:
                return False
        return True

    def __ne__(self, other: Self) -> Bool:
        """`operator!=` (`feature.h:149-151`)."""
        return not (self == other)

    def get_hash(self) -> UInt64:
        """`GetHash()` (`feature.h:161-163`):

            MultiHash(TVecHash<TBinarySplit>()(Splits), VecCityHash(CatFeatures))

        `MultiHash` of two folds from the tail, so the CAT half is hashed
        first and the SPLIT half is XORed in on the left -- and the split
        half is an `int`, so it arrives sign-extended.
        """
        var h = vec_city_hash_u32(self.cat_features)
        return combine_hashes(h, sign_extend_i32(t_vec_hash_splits(self.splits)))

    def less(self, other: Self) -> Bool:
        """`operator<` (`feature.h:165-167`), `std::tie(Splits,
        CatFeatures)`: lexicographic on the split vector first, and only
        on a full tie does the cat vector decide. A `std::vector`
        comparison is `std::lexicographical_compare`, so a PREFIX sorts
        before its extension."""
        var ns = len(self.splits)
        var nos = len(other.splits)
        var n = ns if ns < nos else nos
        for i in range(n):
            if binary_split_less(self.splits[i], other.splits[i]):
                return True
            if binary_split_less(other.splits[i], self.splits[i]):
                return False
        if ns != nos:
            return ns < nos
        var nc = len(self.cat_features)
        var noc = len(other.cat_features)
        var m = nc if nc < noc else noc
        for i in range(m):
            if self.cat_features[i] < other.cat_features[i]:
                return True
            if other.cat_features[i] < self.cat_features[i]:
                return False
        return nc < noc

    def is_subset(self, other: Self) -> Bool:
        """`IsSubset` (`feature.h:169-171`): `NCB::IsSubset` on BOTH
        vectors, which is `std::includes(set, subset)` over sorted ranges
        (`helpers/set.h:7-10`).

        Their argument order reads backwards at the call site --
        `NCB::IsSubset(Splits, other.Splits)` asks whether OUR splits are
        contained in THEIRS -- so `a.IsSubset(b)` is "a is a subset of b".
        """
        var i = 0
        for k in range(len(self.splits)):
            while i < len(other.splits) and binary_split_less(
                other.splits[i], self.splits[k]
            ):
                i += 1
            if i == len(other.splits) or binary_split_less(
                self.splits[k], other.splits[i]
            ):
                return False
            i += 1
        var j = 0
        for k in range(len(self.cat_features)):
            while (
                j < len(other.cat_features)
                and other.cat_features[j] < self.cat_features[k]
            ):
                j += 1
            if (
                j == len(other.cat_features)
                or self.cat_features[k] < other.cat_features[j]
            ):
                return False
            j += 1
        return True

    # -- the scalar answers (`feature.h:91-93`, `:153-183`) ---------------

    def is_simple(self) -> Bool:
        """`IsSimple()` (`feature.h:91-93`): exactly ONE member in total.
        A simple CTR's tensor is one cat feature and no splits; a
        one-split, no-cat tensor is also "simple" by this predicate, which
        is why `IsTreeCtr` is `IsCtr && !IsSimple` and not a size test."""
        return (len(self.splits) + len(self.cat_features)) == 1

    def is_empty(self) -> Bool:
        """`IsEmpty()` (`feature.h:153-155`)."""
        return len(self.cat_features) == 0 and len(self.splits) == 0

    def size(self) -> Int:
        """`Size()` (`feature.h:157-159`), the plain sum. NOT the
        complexity."""
        return len(self.cat_features) + len(self.splits)

    def get_complexity(self) -> Int:
        """`GetComplexity()` (`feature.h:181-183`):

            CatFeatures.size() + std::min<ui64>(Splits.size(), 1)

        ALL THE SPLITS TOGETHER COUNT ONE. A depth-6 tree's six splits
        crossed with one categorical feature has complexity 2, and is
        still a legal base tensor at their default `MaxTensorComplexity`
        of 4. Reading this as `Size()` would refuse almost every tree CTR
        they build.
        """
        var s = len(self.splits)
        return len(self.cat_features) + (1 if s > 0 else 0)

    def get_splits(self) -> List[TBinarySplit]:
        """`GetSplits()` (`feature.h:173-175`). A copy; theirs is a const
        reference and Mojo has no equivalent for a returned field."""
        return self.splits.copy()

    def get_cat_features(self) -> List[UInt32]:
        """`GetCatFeatures()` (`feature.h:177-179`)."""
        return self.cat_features.copy()


# --- the two feature-manager predicates (`binarizations_manager.h:65-72`)


def is_tree_ctrs_enabled(
    has_cat_features: Bool, max_tensor_complexity: Int
) -> Bool:
    """`TBinarizedFeaturesManager::IsTreeCtrsEnabled()`
    (`binarizations_manager.h:65-68`):

        !DataProviderCatFeatureIdToFeatureManagerId.empty()
        && (CatFeatureOptions.MaxTensorComplexity > 1)

    **This is the switch that keeps the whole block off in this port.**
    `TCatFeatureParams.check()` pins `max_ctr_complexity` at 1, so the
    second conjunct is false on every fit `train()` can run today.
    """
    return has_cat_features and max_tensor_complexity > 1


def use_as_base_tensor_for_tree_ctr(
    tensor: TFeatureTensor, max_tensor_complexity: Int
) -> Bool:
    """`UseAsBaseTensorForTreeCtr` (`binarizations_manager.h:70-72`):
    `tensor.GetComplexity() < MaxTensorComplexity`.

    STRICTLY less, and against `GetComplexity()` -- so at their default of
    4 a base may already carry 3 categorical features plus any number of
    splits, and the tensor built from it by adding a fourth cat feature
    has complexity 4 and can no longer be a base itself.
    """
    return tensor.get_complexity() < max_tensor_complexity


# --- `TBatchFeatureTensorBuilder` ---------------------------------------


trait TFeatureTensorVisitor:
    """`TBatchFeatureTensorBuilder::TFeatureTensorVisitor`
    (`batch_feature_tensor_builder.h:10-11`), which is
    `std::function<void(const TFeatureTensor&, TCtrBinBuilder&)>`.

    Their one implementation wraps `TCtrFromTensorCalcer`
    (`tree_ctrs.cpp:500-502`), which is NOT ported. A trait stands in for
    the `std::function` because rule 4 forbids dynamic trait objects here
    and a compile-time parameter is the same dispatch their template would
    have produced anyway.
    """

    def visit(
        mut self, tensor: TFeatureTensor, mut bin_builder: TCtrBinBuilder
    ) raises:
        ...


def _set_indices(mut builder: TCtrBinBuilder, base_tensor_indices: List[UInt32]):
    """`TCtrBinBuilder::SetIndices` for the learn-only case
    (`ctr_bins_builder.h:32-52`) followed by `Reset` (`:193-197`).

    A free function, not a method, because this lane must not edit
    `gbdt/ctrs/ctr_bins_builder.mojo` -- see DEVIATION 118. The base
    tensor's indices ALREADY CARRY their segment-start flags in bit 31
    (they come out of a previous `UpdateBordersMask`), which is why they
    are adopted verbatim rather than validated the way the builder's
    permutation constructor validates a raw order.
    """
    var n = len(base_tensor_indices)
    builder.indices = base_tensor_indices.copy()
    builder.learn_size = n
    builder.bins = List[UInt32]()
    builder.current_bins = List[UInt32]()
    for _ in range(n):
        builder.bins.append(UInt32(0))
        builder.current_bins.append(UInt32(0))


struct TBatchFeatureTensorBuilder(Movable):
    """Their `TBatchFeatureTensorBuilder`
    (`batch_feature_tensor_builder.h:8-37`).

    `FeaturesManager` and `CatFeatures` are held by const reference there;
    here the two things actually read off them -- the per-feature
    categorical codes (`CatFeatures.GetFeatureGpu/Cpu(catFeatureId)`) and
    `FeaturesManager.GetBinCount(catFeatureId)` -- are passed in directly
    and indexed BY FEATURE ID, which is how both of their accessors are
    keyed. DEVIATION 118.
    """

    var cat_feature_bins: List[List[UInt32]]
    """`CatFeatures.GetFeatureCpu(id)`: the dense category code of every
    row, per categorical feature id."""

    var cat_feature_bin_counts: List[Int]
    """`FeaturesManager.GetBinCount(id)`, the `uniqueValues` that decides
    how many bits `ReorderBins` sorts."""

    var tensor_builder_streams: Int
    """`TensorBuilderStreams`, their
    `PackSizeEstimators[deviceId]->GetStreamCountForCtrCalculation()`
    (`tree_ctrs.cpp:481`). The BATCH WIDTH here; DEVIATION 117."""

    var builder_streams: List[Int]
    """`BuilderStreams`. Slot indices, not streams. Grown, never shrunk."""

    var ctr_bin_builders: List[TCtrBinBuilder]
    """`CtrBinBuilders`, one per slot, REUSED across batches and across
    calls."""

    def __init__(
        out self,
        var cat_feature_bins: List[List[UInt32]],
        var cat_feature_bin_counts: List[Int],
        tensor_builder_streams: Int,
    ):
        self.cat_feature_bins = cat_feature_bins^
        self.cat_feature_bin_counts = cat_feature_bin_counts^
        self.tensor_builder_streams = tensor_builder_streams
        self.builder_streams = List[Int]()
        self.ctr_bin_builders = List[TCtrBinBuilder]()

    def request_stream(mut self, features_to_build: Int) -> Int:
        """`RequestStream` (`batch_feature_tensor_builder.cpp:67-77`).

            buildStreams = min(TensorBuilderStreams, featuresToBuild)
            for i = BuilderStreams.size(); i < buildStreams; ++i:
                BuilderStreams.push_back(...); CtrBinBuilders.push_back(...)
            return buildStreams

        The loop starts at the CURRENT size, so a later call with more
        features extends the pool and a later call with fewer leaves it
        alone -- the pool only ever grows, and the returned width can be
        smaller than the pool.
        """
        var build_streams = self.tensor_builder_streams
        if features_to_build < build_streams:
            build_streams = features_to_build
        for _ in range(len(self.builder_streams), build_streams):
            self.builder_streams.append(len(self.builder_streams))
            self.ctr_bin_builders.append(TCtrBinBuilder(0))
        return build_streams

    def visit_ctr_bin_builders[
        V: TFeatureTensorVisitor
    ](
        mut self,
        base_tensor_indices: List[UInt32],
        base_tensor: TFeatureTensor,
        cat_feature_ids: List[UInt32],
        mut visitor: V,
    ) raises:
        """`VisitCtrBinBuilders`
        (`batch_feature_tensor_builder.cpp:10-65`), branch for branch.

        The `currentBins` precomputation at their `:15-19` is dropped --
        DEVIATION 118 -- because the builder recomputes the same array
        from the indices it was just reset to.

        THE TWO INNER LOOPS ARE NOT MERGED. Their comment at `:24` says
        why ("do not merge with second part. ctrBinBuilder should be async
        wrt host"): the first loop submits, the second consumes. Merging
        them is a one-line change that costs the whole batch's overlap and
        changes no answer, which is the shape of deviation nothing catches.
        """
        var build_streams = self.request_stream(len(cat_feature_ids))

        var i = 0
        while i < len(cat_feature_ids):
            # submit build tensors (`:26-52`)
            for j in range(build_streams):
                var feature_index = i + j
                if feature_index < len(cat_feature_ids):
                    var cat_feature_id = Int(cat_feature_ids[feature_index])
                    _set_indices(
                        self.ctr_bin_builders[j], base_tensor_indices
                    )
                    self.ctr_bin_builders[j].add_cat_feature_bins(
                        self.cat_feature_bins[cat_feature_id],
                        self.cat_feature_bin_counts[cat_feature_id],
                    )

            # visit tensors (`:54-64`)
            for j in range(build_streams):
                var feature_index = i + j
                if feature_index < len(cat_feature_ids):
                    var cat_feature_id = cat_feature_ids[feature_index]
                    var feature_tensor = base_tensor.copy()
                    feature_tensor.add_cat_feature(cat_feature_id)
                    visitor.visit(feature_tensor, self.ctr_bin_builders[j])

            i += build_streams
