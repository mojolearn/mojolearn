# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gate `TFeatureTensor`'s identity -- canonical form, equality, order,
subset, complexity and HASH -- and the loop that mints one tensor per
categorical feature.

    pixi run mojo run -I . original/feature_tensor_check.mojo

## The oracle

`ORACLE` below is the output of a C++ program whose every line above
`main()` is a TRANSCRIPTION of CatBoost's own source at `54a8143a`,
compiled by clang:

    util/digest/numeric.h:39-93            IntHashImpl, IntHash, CombineHashes
    util/digest/multi.h:6-14               MultiHash
    util/str_stl.h:38-54, 159              THash -> hash -> THashHelper
    catboost/libs/model/hash.h:16-25       TVecHash<T>
    catboost/libs/helpers/hash.h:6-9       VecCityHash
    catboost/libs/helpers/set.h:7-10       NCB::IsSubset
    catboost/cuda/data/feature.h:35-183    TBinarySplit, TFeatureTensor

linked against `tools/cityhash_oracle/city.{h,cpp}`, which are their own
`util/digest/city.{h,cpp}` byte for byte. So this is a second
implementation, in a different language, through a different compiler,
and the comparison is evidence rather than a restatement of the port.
Their `city.h` states outright that its results DIFFER from mainline
CityHash, so a public CityHash vector would be the wrong oracle; theirs is
the only ground truth and `pixi run check-cityhash` already gates the
CityHash half cell for cell.

Reproduce (the generator lives outside this lane's two files; it belongs
under `tools/feature_tensor_oracle/` the moment someone owns `tools/`):

    c++ -std=c++17 -O2 -I tools/cityhash_oracle \
        <generator>.cpp tools/cityhash_oracle/city.cpp -o /tmp/tensor_oracle
    /tmp/tensor_oracle

**Every fixture's INSERTION ORDER is read out of the oracle text, not
restated here**, so the two sides cannot drift apart in what they built.

## The gates

1. **SPLIT HASH** -- `TBinarySplit::GetHash()` for all 25 distinct splits
   the fixtures mention, including `0x7fffffff` and `0x80000001`.
2. **TENSOR** -- for all 52 fixtures, built in the oracle's own insertion
   order: canonical splits, canonical cat features, `TVecHash` of the
   split half (as a SIGNED int), `VecCityHash` of the cat half, the
   combined `GetHash`, `Size`, `GetComplexity`, `IsSimple`.
3. **ORDER INDEPENDENCE** -- six different build routes per fixture
   (forward, reverse, interleaved, the bulk `AddBinarySplit(TVector)` /
   `AddCatFeature(TVector)` overloads, `AddTensor` from singletons, and
   every element inserted twice in a rotated order). All six must be `==`
   and hash-identical to each other AND to the oracle's number. Their
   canonicalisation is order-INDEPENDENT and this is what says so;
   fixtures 8, 9, 10 and 11 are the same tensor built four ways in the
   ORACLE too, so their agreement is CatBoost's statement, not ours.
4. **PAIRS** -- `==`, `<` and `IsSubset` over all 52 x 52 ordered pairs
   against the oracle's bitmasks, plus three properties the oracle cannot
   be wrong about on its own: trichotomy (exactly one of `a<b`, `b<a`,
   `a==b`), `a==b` implies equal hashes, and `a<b` implies different
   hashes are permitted but equality is not.
5. **MERGE** -- `AddTensor` on 52 pairs.
6. **COLLISIONS** -- 800 pairwise-DIFFERENT tensors. All 800 hashes must
   be distinct, and their count, distinct count, XOR and SUM must match
   the oracle's. **A hash that returns a constant passes every equality
   test ever written and dies here.** So does one that drops either half:
   the set varies the split feature, the split bin, the split TYPE and the
   categorical feature independently.
7. **PREDICATES** -- `IsTreeCtrsEnabled` and `UseAsBaseTensorForTreeCtr`,
   both sides of both switches, including the `<` vs `<=` boundary at
   complexity == MaxTensorComplexity.
8. **BATCH BUILDER** -- `VisitCtrBinBuilders` at four batch widths.

## Gate 8, and the trap it exists for

`VisitCtrBinBuilders` GROUPS its features: `buildStreams` of them share a
pass, and feature `i` lands in builder slot `i % buildStreams`. A fixture
with `buildStreams` features or fewer runs ONE group and can see none of
that -- not a slot reused for a second feature, not a stale builder
carried between groups, not an off-by-one in the `i + j` addressing. This
repository has been bitten by exactly that shape five times in two days,
most recently by a fixture whose whole signal sat in the first group of a
policy.

So the fixture is 7 categorical features at width 3: groups (11,12,13),
(14,15,16), (17), and slot 0 serves features 11, 14 AND 17. Every
feature has a DIFFERENT cardinality (3..9) and a different code pattern,
so a slot that serves the wrong feature, or one whose `SetIndices` reset
is skipped, produces a different ordering rather than the same one.

Gated per feature: the tensor visited (against the oracle's hash for the
first three, which are fixtures 46, 47 and 48), and the bin builder's
resulting row ordering including its segment-start flags, against a
grouping computed here from the base segmentation and the feature's codes.
Then the whole thing is re-run at widths 1, 7 and 100: the batch width
changes WHEN work happens and must change NOTHING that comes out.

The base ordering is 23 rows -- prime, so no power-of-two coincidence can
make two offsets agree -- in four segments of sizes 5, 7, 3 and 8, in a
scrambled row order, so a check that ignored placement and counted totals
would pass while the rows sat anywhere.

## SABOTAGE: which planted defect moved which gate

Run 2026-08-21, one sabotage per MECHANISM, each applied to the library
alone, the check untouched, then reverted.

| sabotage in `batch_feature_tensor_builder.mojo` | gates that went red |
|---|---|
| `GetHash` returns a constant | 2 (52), 5 (52), 6 (**319,600 colliding pairs**), 8 (3) |
| `CombineHashes` mixes the RIGHT argument instead of the left | 1, 2, 5, 6, 8 |
| `THash<int>` zero-extends instead of sign-extending | 2 (11), 5, 6 |
| `TVecHash` folds each split hash's HIGH half, not its low | 2 (49), 5 (52), 6 (1,920 collisions), 8 |
| `VecCityHash` assembles the bytes big-endian | 2, 5, 6, 8 |
| `SortUniqueSplits` stops SORTING | 2, 3, 4, 5 |
| `SortUniqueSplits` stops DEDUPLICATING | 2, 3, 4, 5 |
| `_as_u32` reads the fields SIGNED (DEVIATION 116 dropped) | 1, 2, 4, 5 |
| `GetComplexity` returns `Size` | 2, 5, 7 |
| `IsSubset` asks the containment the other way round | 4, 5 |
| `operator<` compares the cat features first | 4 (1,404) |
| the batch loop never re-runs `SetIndices` (stale slot) | 8 only |
| the cat column is indexed by BATCH POSITION, not feature id | 8 only |
| the outer loop advances by 1 instead of the batch width | 8 only |

**Two sabotages moved NOTHING, and both are honest zeroes rather than
holes:**

* *`TVecHash` accumulates in `ui64` and truncates once at the end.* This
  cannot be caught, because it is the same function: truncation commutes
  with a fold built out of `*` and `+`, so `(984121*res + h) mod 2^32`
  computed stepwise and computed at the end agree bit for bit. The quirk
  that DOES matter -- that each split's 64-bit hash is truncated at all --
  is the `S4'` row above, which moves four gates.
* *The two inner loops of `VisitCtrBinBuilders` merged into one.* Their
  `:24` comment forbids it ("do not merge with second part. ctrBinBuilder
  should be async wrt host"), and at one queue it changes no output, so no
  check here or anywhere can see it. It is a LATENCY deviation, kept apart
  on their authority, and this file cannot be cited as coverage for it.

The first three batch sabotages are the point of the 7-features-at-width-3
fixture: each is invisible in the first group and every one of them is
caught in the second.

## What the adversarial fixture found

The `0x80000001` / `0xffffffff` split in fixtures 41 and 42 was written to
guard DEVIATION 116, and on its first run it failed for a different
reason: `UInt64(_as_u32(v))` SIGN-EXTENDS in Mojo 1.0 when `_as_u32`'s
body is a same-width `Int32 -> UInt32` cast. Nothing else in this file
moved -- the canonical form, the comparator, `IsSubset` and every scalar
were green, because they compare two values mangled identically. See the
HAZARD block in the library's module docstring.
"""

from gbdt.ctrs.ctr_bins_builder import TCtrBinBuilder
from gbdt.ctrs.index_wrapper import index_of, is_segment_start, with_mask
from gbdt.methods.batch_feature_tensor_builder import (
    TBatchFeatureTensorBuilder,
    TFeatureTensor,
    TFeatureTensorVisitor,
    binary_split_hash,
    is_tree_ctrs_enabled,
    t_vec_hash_splits,
    use_as_base_tensor_for_tree_ctr,
    vec_city_hash_u32,
)
from gbdt.models.oblivious_model import TBinarySplit


comptime ORACLE = """TENSORS 52
T 0 0 0 0 0 1988712 9ae16a3b2f90404f feda278c80fac35c 0 0 0
T 1 1 3 1 0 0 1 3 1 0 0 1767327395 9ae16a3b2f90404f feda278ce9b3dd97 1 1 1
T 2 0 1 7 0 1 7 1988712 cadbb131f0b06e45 b45e8c4e3017c7cc 1 1 1
T 3 1 3 1 0 1 7 1 3 1 0 1 7 1767327395 cadbb131f0b06e45 b45e8c4e595ed907 2 2 0
T 4 1 3 1 1 1 7 1 3 1 1 1 7 553795298 cadbb131f0b06e45 b45e8c4e110ba146 2 2 0
T 5 1 3 2 0 1 7 1 3 2 0 1 7 1093674609 cadbb131f0b06e45 b45e8c4e7139b9d5 2 2 0
T 6 1 4 1 0 1 7 1 4 1 0 1 7 1767327396 cadbb131f0b06e45 b45e8c4e595ed900 2 2 0
T 7 1 3 1 0 1 8 1 3 1 0 1 8 1767327395 007a8729113cc6dc 66d2718b6ce4e122 2 2 0
T 8 2 3 1 0 4 2 1 2 7 8 2 3 1 0 4 2 1 2 7 8 -1067636315 da551c0d4285d174 8e298a37b2d0d395 4 3 0
T 9 2 4 2 1 3 1 0 2 8 7 2 3 1 0 4 2 1 2 7 8 -1067636315 da551c0d4285d174 8e298a37b2d0d395 4 3 0
T 10 4 4 2 1 3 1 0 4 2 1 3 1 0 5 8 7 7 8 8 2 3 1 0 4 2 1 2 7 8 -1067636315 da551c0d4285d174 8e298a37b2d0d395 4 3 0
T 11 2 3 1 0 4 2 1 2 8 7 2 3 1 0 4 2 1 2 7 8 -1067636315 da551c0d4285d174 8e298a37b2d0d395 4 3 0
T 12 1 3 1 0 1 1000 1 3 1 0 1 1000 1767327395 0fe94aa1dbbd1673 80006bc36f481330 2 2 0
T 13 1 3 1 0 2 1000 1007 1 3 1 0 2 1000 1007 1767327395 9a530a030b7bef5a e5e05b891a4852de 3 3 0
T 14 1 3 1 0 3 1000 1007 1014 1 3 1 0 3 1000 1007 1014 1767327395 3a9fab51fa733b8f fa125c166d9264cd 4 4 0
T 15 1 3 1 0 4 1000 1007 1014 1021 1 3 1 0 4 1000 1007 1014 1021 1767327395 7010437f17c0edfb f83c027a2a98c64f 5 5 0
T 16 1 3 1 0 5 1000 1007 1014 1021 1028 1 3 1 0 5 1000 1007 1014 1021 1028 1767327395 e43295d6cd3ab98a f517f8a0313cfc30 6 6 0
T 17 1 3 1 0 6 1000 1007 1014 1021 1028 1035 1 3 1 0 6 1000 1007 1014 1021 1028 1035 1767327395 399ea48cf8c6409e ba828d80ec61a592 7 7 0
T 18 1 3 1 0 7 1000 1007 1014 1021 1028 1035 1042 1 3 1 0 7 1000 1007 1014 1021 1028 1035 1042 1767327395 a0f26d889f14fd77 e3f6f11ec5d638dc 8 8 0
T 19 1 3 1 0 8 1000 1007 1014 1021 1028 1035 1042 1049 1 3 1 0 8 1000 1007 1014 1021 1028 1035 1042 1049 1767327395 aaff6e8b008fa47f 57527d2492c60bf7 9 9 0
T 20 1 3 1 0 9 1000 1007 1014 1021 1028 1035 1042 1049 1056 1 3 1 0 9 1000 1007 1014 1021 1028 1035 1042 1049 1056 1767327395 ff361b08f9f467f0 b0adfac11d92cd17 10 10 0
T 21 1 3 1 0 10 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1 3 1 0 10 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1767327395 c905fb82930d68b2 3cb8e085a74c68fe 11 11 0
T 22 1 3 1 0 11 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1 3 1 0 11 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1767327395 f496c0ea8eeca469 c3d2ce27e9632e13 12 12 0
T 23 1 3 1 0 12 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1 3 1 0 12 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1767327395 9b6bbb4982b0a309 16e1a7b363525776 13 13 0
T 24 1 3 1 0 13 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1084 1 3 1 0 13 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1084 1767327395 17aa77fc85ba5bf4 72f81f2d22b47075 14 14 0
T 25 1 3 1 0 14 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1084 1091 1 3 1 0 14 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1084 1091 1767327395 977733e58a25a7c8 2adcb4e96bed9389 15 15 0
T 26 1 3 1 0 15 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1084 1091 1098 1 3 1 0 15 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1084 1091 1098 1767327395 d710f313884a8309 efb9bff3e37f56bd 16 16 0
T 27 1 3 1 0 16 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1084 1091 1098 1105 1 3 1 0 16 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1084 1091 1098 1105 1767327395 49d1c8f2258d2599 62a4cc9c24587893 17 17 0
T 28 1 3 1 0 17 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1084 1091 1098 1105 1112 1 3 1 0 17 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1084 1091 1098 1105 1112 1767327395 5dd6a236a5c48e29 bc1843900ff3aae9 18 18 0
T 29 1 3 1 0 18 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1084 1091 1098 1105 1112 1119 1 3 1 0 18 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1084 1091 1098 1105 1112 1119 1767327395 084562d69a9d258f 760d3461fa92663f 19 19 0
T 30 1 3 1 0 19 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1084 1091 1098 1105 1112 1119 1126 1 3 1 0 19 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1084 1091 1098 1105 1112 1119 1126 1767327395 173767a222285203 bb9965a689d75f5e 20 20 0
T 31 1 3 1 0 20 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1084 1091 1098 1105 1112 1119 1126 1133 1 3 1 0 20 1000 1007 1014 1021 1028 1035 1042 1049 1056 1063 1070 1077 1084 1091 1098 1105 1112 1119 1126 1133 1767327395 299a8fd56769c471 dfadc34b93bbdb38 21 21 0
T 32 1 3 1 0 33 2000 2003 2006 2009 2012 2015 2018 2021 2024 2027 2030 2033 2036 2039 2042 2045 2048 2051 2054 2057 2060 2063 2066 2069 2072 2075 2078 2081 2084 2087 2090 2093 2096 1 3 1 0 33 2000 2003 2006 2009 2012 2015 2018 2021 2024 2027 2030 2033 2036 2039 2042 2045 2048 2051 2054 2057 2060 2063 2066 2069 2072 2075 2078 2081 2084 2087 2090 2093 2096 1767327395 7e05679259a55562 4caf2394d139249d 34 34 0
T 33 1 10 1 0 1 5 1 10 1 0 1 5 1767327386 1fa7bf968bc45deb 31a83c5616a067cc 2 2 0
T 34 2 10 1 0 11 3 1 1 5 2 10 1 0 11 3 1 1 5 1432129259 1fa7bf968bc45deb 31a83c562aabafbd 3 2 0
T 35 3 10 1 0 11 3 1 12 5 0 1 5 3 10 1 0 11 3 1 12 5 0 1 5 -1375384847 1fa7bf968bc45deb ce57c3a9d1f26ba7 4 2 0
T 36 4 10 1 0 11 3 1 12 5 0 13 7 1 1 5 4 10 1 0 11 3 1 12 5 0 13 7 1 1 5 699240390 1fa7bf968bc45deb 31a83c56565aae90 5 2 0
T 37 5 10 1 0 11 3 1 12 5 0 13 7 1 14 9 0 1 5 5 10 1 0 11 3 1 12 5 0 13 7 1 14 9 0 1 5 100824231 1fa7bf968bc45deb 31a83c5679f555f1 6 2 0
T 38 6 10 1 0 11 3 1 12 5 0 13 7 1 14 9 0 15 11 1 1 5 6 10 1 0 11 3 1 12 5 0 13 7 1 14 9 0 15 11 1 1 5 804715721 1fa7bf968bc45deb 31a83c565001dd9f 7 2 0
T 39 7 10 1 0 11 3 1 12 5 0 13 7 1 14 9 0 15 11 1 16 13 0 1 5 7 10 1 0 11 3 1 12 5 0 13 7 1 14 9 0 15 11 1 16 13 0 1 5 -1358050540 1fa7bf968bc45deb ce57c3a9d0faea42 8 2 0
T 40 8 10 1 0 11 3 1 12 5 0 13 7 1 14 9 0 15 11 1 16 13 0 17 15 1 1 5 8 10 1 0 11 3 1 12 5 0 13 7 1 14 9 0 15 11 1 16 13 0 17 15 1 1 5 -715832197 1fa7bf968bc45deb ce57c3a9aaa2652d 9 2 0
T 41 2 2147483649 4294967295 1 2147483647 0 0 3 2147483648 4294967295 1 2 2147483647 0 0 2147483649 4294967295 1 3 1 2147483648 4294967295 -948720840 9fbfd35cea864c76 bc80e1a0d7780e11 5 4 0
T 42 2 2147483647 0 0 2147483649 4294967295 1 3 4294967295 1 2147483648 2 2147483647 0 0 2147483649 4294967295 1 3 1 2147483648 4294967295 -948720840 9fbfd35cea864c76 bc80e1a0d7780e11 5 4 0
T 43 2 9 9 1 9 9 0 0 2 9 9 0 9 9 1 0 -1882274050 9ae16a3b2f90404f 0125d8730f2a5dca 2 1 0
T 44 2 9 9 0 9 8 0 0 2 9 8 0 9 9 0 0 1203509089 9ae16a3b2f90404f feda278cc7588c55 2 1 0
T 45 3 1 5 1 2 3 1 3 7 1 0 3 1 5 1 2 3 1 3 7 1 0 713979794 9ae16a3b2f90404f feda278caa6aeca6 3 1 0
T 46 3 1 5 1 2 3 1 3 7 1 1 11 3 1 5 1 2 3 1 3 7 1 1 11 713979794 ba17ffa84abc4b1a bd9a8830f580724e 4 2 0
T 47 3 1 5 1 2 3 1 3 7 1 1 12 3 1 5 1 2 3 1 3 7 1 1 12 713979794 d13d6a61d145fe49 dbe1efd379c0bc8f 4 2 0
T 48 3 1 5 1 2 3 1 3 7 1 1 13 3 1 5 1 2 3 1 3 7 1 1 13 713979794 8fa0fa4bc32a838d 3edc31cfda0468a3 4 2 0
T 49 5 1 1 1 2 1 1 3 1 1 4 1 1 5 1 1 3 20 21 22 5 1 1 1 2 1 1 3 1 1 4 1 1 5 1 1 3 20 21 22 -1818781698 06b89aae759a42da 93e83b92a778c7a4 8 4 0
T 50 5 1 1 1 2 1 1 3 1 1 4 1 1 5 1 1 2 20 21 5 1 1 1 2 1 1 3 1 1 4 1 1 5 1 1 2 20 21 -1818781698 dbeb5c66a8115a3a f0cee34dd73b112d 7 3 0
T 51 0 4 20 21 22 23 0 4 20 21 22 23 1988712 b2cb678dd7f7e992 bb3fbf05c1ef6df7 4 4 0
SPLITS 25
S 1 1 1 5ce9ee4e72c6efb8
S 1 5 1 d63837bc0bee3c1f
S 2 1 1 5ce9ee4e72c6efbb
S 2 3 1 b7b3995d60cd6ea8
S 3 1 0 db6a382fbb1bf77b
S 3 1 1 5ce9ee4e72c6efba
S 3 2 0 2238fe8a92f4d749
S 3 7 1 e9a3dcadf1351c13
S 4 1 0 db6a382fbb1bf77c
S 4 1 1 5ce9ee4e72c6efbd
S 4 2 1 541ac1e4cb46e35a
S 5 1 1 5ce9ee4e72c6efbc
S 9 8 0 e4ad8230cb242afb
S 9 9 0 3e8d63c99b845996
S 9 9 1 ba53c6ac425d3cb0
S 10 1 0 db6a382fbb1bf772
S 11 3 1 b7b3995d60cd6ea1
S 12 5 0 965526e5da68cc9e
S 13 7 1 e9a3dcadf1351c1d
S 14 9 0 3e8d63c99b845991
S 15 11 1 4b96cd41f9cf679a
S 16 13 0 6310240355555e53
S 17 15 1 38998821bdeabd07
S 2147483647 0 0 b1d7c7bd5469c284
S 2147483649 4294967295 1 f965957793580bec
PAIRS 52
P 0 1000000000000 effffffffffff fffffffffffff 
P 1 2000000000000 8ffffffffff10 a8ffffff10000 
P 2 4000000000000 affffffffffff c7f0000000000 
P 3 8000000000000 0ffffffffff10 80f0000000000 
P 4 0100000000000 06000000eff10 0100000000000 
P 5 0200000000000 04000000eff10 0200000000000 
P 6 0400000000000 00000000eff10 0400000000000 
P 7 0800000000000 07fffffffff10 08f0000000000 
P 8 00f0000000000 07000000eff10 00f0000000000 
P 9 00f0000000000 07000000eff10 00f0000000000 
P 10 00f0000000000 07000000eff10 00f0000000000 
P 11 00f0000000000 07000000eff10 00f0000000000 
P 12 0001000000000 07fefffffff10 000fffff00000 
P 13 0002000000000 07fcfffffff10 000effff00000 
P 14 0004000000000 07f8fffffff10 000cffff00000 
P 15 0008000000000 07f0fffffff10 0008ffff00000 
P 16 0000100000000 07f0effffff10 0000ffff00000 
P 17 0000200000000 07f0cffffff10 0000efff00000 
P 18 0000400000000 07f08ffffff10 0000cfff00000 
P 19 0000800000000 07f00ffffff10 00008fff00000 
P 20 0000010000000 07f00efffff10 00000fff00000 
P 21 0000020000000 07f00cfffff10 00000eff00000 
P 22 0000040000000 07f008fffff10 00000cff00000 
P 23 0000080000000 07f000fffff10 000008ff00000 
P 24 0000001000000 07f000effff10 000000ff00000 
P 25 0000002000000 07f000cffff10 000000ef00000 
P 26 0000004000000 07f0008ffff10 000000cf00000 
P 27 0000008000000 07f0000ffff10 0000008f00000 
P 28 0000000100000 07f0000efff10 0000000f00000 
P 29 0000000200000 07f0000cfff10 0000000e00000 
P 30 0000000400000 07f00008fff10 0000000c00000 
P 31 0000000800000 07f00000fff10 0000000800000 
P 32 0000000010000 07f00000eff10 0000000010000 
P 33 0000000020000 00000000cf700 00000000ef100 
P 34 0000000040000 000000008f700 00000000cf100 
P 35 0000000080000 000000000f700 000000008f100 
P 36 0000000001000 000000000e700 000000000f100 
P 37 0000000002000 000000000c700 000000000e100 
P 38 0000000004000 0000000008700 000000000c100 
P 39 0000000008000 0000000000700 0000000008100 
P 40 0000000000100 0000000000600 0000000000100 
P 41 0000000000600 0000000000000 0000000000600 
P 42 0000000000600 0000000000000 0000000000600 
P 43 0000000000800 00000000ef700 0000000000800 
P 44 0000000000010 00000000eff00 0000000000010 
P 45 0000000000020 affffffffffd1 00000000000e1 
P 46 0000000000040 affffffffff91 0000000000040 
P 47 0000000000080 affffffffff11 0000000000080 
P 48 0000000000001 affffffffff10 0000000000001 
P 49 0000000000002 afffffffffff1 0000000000002 
P 50 0000000000004 afffffffffff3 0000000000006 
P 51 0000000000008 afffffffffff7 0000000000008 
MERGES 52
M 0 3 b45e8c4e595ed907 2 2
M 1 10 8e298a37b2d0d395 4 3
M 2 17 96e1146d98e7e8ba 8 8
M 3 24 2025bd219d43db36 15 15
M 4 31 e169f30eada42226 23 22
M 5 38 170d28e619415c70 9 3
M 6 45 b45e8c4e4049415a 5 2
M 7 0 66d2718b6ce4e122 2 2
M 8 7 8e298a37b2d0d395 4 3
M 9 14 9ffea10052e4c701 7 6
M 10 21 f48887bef13a69d7 14 13
M 11 28 c028a9d2f159570f 21 20
M 12 35 203005fb3d3d36f7 6 3
M 13 42 8d12b10e455c3c7f 8 6
M 14 49 1c5e4085263bd2e2 12 7
M 15 4 0b36d967b5730cc8 7 6
M 16 11 1e32bf17dbe6e3aa 9 8
M 17 18 e3f6f11ec5d638dc 8 8
M 18 25 2adcb4e96bed9389 15 15
M 19 32 909f21b00b2c2cfc 42 42
M 20 39 9a96d80ef3458514 18 11
M 21 46 b6921ed79aedd9bf 15 12
M 22 1 c3d2ce27e9632e13 12 12
M 23 8 d0629bc2e7a5ff3c 16 15
M 24 15 72f81f2d22b47075 14 14
M 25 22 2adcb4e96bed9389 15 15
M 26 29 760d3461fa92663f 19 19
M 27 36 ff23a1fa95bb3c62 22 18
M 28 43 bc1843905bddb583 20 18
M 29 50 17f1d3d2c5012aa0 26 21
M 30 5 c179b8c3b6ff4b58 22 21
M 31 12 dfadc34b93bbdb38 21 21
M 32 19 909f21b00b2c2cfc 42 42
M 33 26 b9e510dd4da707fe 18 17
M 34 33 31a83c562aabafbd 3 2
M 35 40 ce57c3a9aaa2652d 9 2
M 36 47 a607faaf21041d42 9 3
M 37 2 170d28e67f8024b5 7 3
M 38 9 12903ac1b56103ea 11 4
M 39 16 e86c72610dbed9de 14 7
M 40 23 703e0c268f0ebd36 22 14
M 41 30 7b7bdcee568394a5 25 23
M 42 37 28ebcf84158f6a47 11 5
M 43 44 0125d8730aae6e7d 3 1
M 44 51 bb3fbf05864d22fe 6 5
M 45 6 b45e8c4e4049415a 5 2
M 46 13 4c6c6e21f9f9a7c8 7 4
M 47 20 9313c94e11c5ae6f 14 11
M 48 27 0f3a6345f7277543 21 18
M 49 34 da479c3c2e0b5a40 11 5
M 50 41 b9d982114fa17701 12 6
M 51 48 26e4bb93a057c645 8 6
ENUM 800 800 0000000000383ea8 e194a8b38618cf98
"""


# --- oracle parsing ------------------------------------------------------


def _hex_digit(b: UInt8) raises -> Int:
    var v = Int(b)
    if v >= 48 and v <= 57:
        return v - 48
    if v >= 97 and v <= 102:
        return v - 97 + 10
    raise Error("bad hex digit: " + String(v))


def _parse_hex_u64(tok: String) raises -> UInt64:
    var v = UInt64(0)
    for b in tok.as_bytes():
        v = (v << 4) | UInt64(_hex_digit(b))
    return v


def _next_str(toks: List[String], mut pos: Int) raises -> String:
    var v = toks[pos]
    pos += 1
    return v


def _next_int(toks: List[String], mut pos: Int) raises -> Int:
    var v = Int(toks[pos])
    pos += 1
    return v


def _next_hex(toks: List[String], mut pos: Int) raises -> UInt64:
    var v = _parse_hex_u64(toks[pos])
    pos += 1
    return v


def _expect(toks: List[String], mut pos: Int, want: String) raises:
    var got = _next_str(toks, pos)
    if got != want:
        raise Error(
            "oracle out of step at token " + String(pos - 1) + ": got '"
            + got + "' want '" + want + "'"
        )


struct Fixture(Copyable, Movable):
    var insert_splits: List[TBinarySplit]
    var insert_cats: List[UInt32]
    var canon_splits: List[TBinarySplit]
    var canon_cats: List[UInt32]
    var vec_hash: Int32
    var city: UInt64
    var hash: UInt64
    var size: Int
    var complexity: Int
    var simple: Bool

    def __init__(out self):
        self.insert_splits = List[TBinarySplit]()
        self.insert_cats = List[UInt32]()
        self.canon_splits = List[TBinarySplit]()
        self.canon_cats = List[UInt32]()
        self.vec_hash = Int32(0)
        self.city = UInt64(0)
        self.hash = UInt64(0)
        self.size = 0
        self.complexity = 0
        self.simple = False


def _split_str(s: TBinarySplit) -> String:
    return (
        "("
        + String(s.feature_id)
        + ","
        + String(s.bin_idx)
        + ","
        + String(s.split_type)
        + ")"
    )


def _splits_equal(a: List[TBinarySplit], b: List[TBinarySplit]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if (
            a[i].feature_id != b[i].feature_id
            or a[i].bin_idx != b[i].bin_idx
            or a[i].split_type != b[i].split_type
        ):
            return False
    return True


def _cats_equal(a: List[UInt32], b: List[UInt32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


# --- the six build routes (gate 3) --------------------------------------


def _build_forward(f: Fixture) raises -> TFeatureTensor:
    """The oracle's own order: every split one at a time, then every cat."""
    var t = TFeatureTensor()
    for i in range(len(f.insert_splits)):
        t.add_binary_split(f.insert_splits[i])
    for i in range(len(f.insert_cats)):
        t.add_cat_feature(f.insert_cats[i])
    return t^


def _build_reverse(f: Fixture) raises -> TFeatureTensor:
    """Cats first, then splits, each reversed."""
    var t = TFeatureTensor()
    for i in range(len(f.insert_cats) - 1, -1, -1):
        t.add_cat_feature(f.insert_cats[i])
    for i in range(len(f.insert_splits) - 1, -1, -1):
        t.add_binary_split(f.insert_splits[i])
    return t^


def _build_interleaved(f: Fixture) raises -> TFeatureTensor:
    """Alternating, from the CANONICAL lists rather than the insertion
    ones, so the two sides do not even share a traversal."""
    var t = TFeatureTensor()
    var n = len(f.canon_splits)
    if len(f.canon_cats) > n:
        n = len(f.canon_cats)
    for k in range(n):
        if k < len(f.canon_cats):
            t.add_cat_feature(f.canon_cats[len(f.canon_cats) - 1 - k])
        if k < len(f.canon_splits):
            t.add_binary_split(f.canon_splits[len(f.canon_splits) - 1 - k])
    return t^


def _build_bulk(f: Fixture) raises -> TFeatureTensor:
    """Their `AddBinarySplit(const TVector<TBinarySplit>&)` and
    `AddCatFeature(const TVector<ui32>&)` overloads, which push every
    element and sort-unique ONCE at the end -- a different code path from
    the single-element ones."""
    var t = TFeatureTensor()
    var ss = List[TBinarySplit]()
    for i in range(len(f.insert_splits) - 1, -1, -1):
        ss.append(f.insert_splits[i])
    var cs = List[UInt32]()
    for i in range(len(f.insert_cats) - 1, -1, -1):
        cs.append(f.insert_cats[i])
    t.add_binary_splits(ss)
    t.add_cat_features(cs)
    return t^


def _build_by_tensor(f: Fixture) raises -> TFeatureTensor:
    """`AddTensor`, folding in one singleton tensor at a time, in reverse."""
    var t = TFeatureTensor()
    for i in range(len(f.insert_cats) - 1, -1, -1):
        var one = TFeatureTensor()
        one.add_cat_feature(f.insert_cats[i])
        t.add_tensor(one)
    for i in range(len(f.insert_splits) - 1, -1, -1):
        var one = TFeatureTensor()
        one.add_binary_split(f.insert_splits[i])
        t.add_tensor(one)
    return t^


def _build_doubled(f: Fixture) raises -> TFeatureTensor:
    """Every element inserted TWICE, rotated by one, so the dedup arm runs
    on every fixture rather than only the three that were written with
    duplicates."""
    var t = TFeatureTensor()
    var ns = len(f.insert_splits)
    for k in range(2 * ns):
        t.add_binary_split(f.insert_splits[(k + 1) % ns if ns > 0 else 0])
    var nc = len(f.insert_cats)
    for k in range(2 * nc):
        t.add_cat_feature(f.insert_cats[(k + 1) % nc if nc > 0 else 0])
    return t^


# --- gate 8's independent grouping --------------------------------------


comptime BATCH_ROWS = 23
comptime BATCH_FIRST_CAT_ID = 11
comptime BATCH_CAT_COUNT = 7


def _base_row_order() -> List[Int]:
    """A scrambled permutation of 0..22. NOT the identity: a check whose
    base order is the identity cannot tell a gather through `Indices` from
    a gather by position."""
    var order: List[Int] = [
        17, 3, 20, 8, 11,
        0, 14, 22, 5, 9, 1, 18,
        12, 6, 21,
        2, 16, 7, 19, 10, 4, 13, 15,
    ]
    return order^


def _base_segment_starts() -> List[Int]:
    """Four segments of sizes 5, 7, 3 and 8 -- unequal on purpose, so a
    reduction that assumed a uniform stride lands in the wrong one."""
    var starts: List[Int] = [0, 5, 12, 15]
    return starts^


def _base_indices() raises -> List[UInt32]:
    var order = _base_row_order()
    var starts = _base_segment_starts()
    var out = List[UInt32]()
    for i in range(len(order)):
        var flag = False
        for k in range(len(starts)):
            if starts[k] == i:
                flag = True
        out.append(with_mask(UInt32(order[i]), flag))
    return out^


def _cat_cardinality(cat_id: Int) -> Int:
    return 3 + (cat_id - BATCH_FIRST_CAT_ID)


def _cat_code(cat_id: Int, row: Int) -> Int:
    var f = cat_id - BATCH_FIRST_CAT_ID
    return (row * (2 * f + 3) + f * 5) % _cat_cardinality(cat_id)


def _cat_columns() -> List[List[UInt32]]:
    """Indexed BY FEATURE ID, as both of their accessors are. Ids below
    `BATCH_FIRST_CAT_ID` are empty and never read; the builder must index
    by id, not by position in `cat_feature_ids`, and an off-by-eleven
    would read one of them."""
    var cols = List[List[UInt32]]()
    for cat_id in range(BATCH_FIRST_CAT_ID + BATCH_CAT_COUNT):
        var col = List[UInt32]()
        if cat_id >= BATCH_FIRST_CAT_ID:
            for row in range(BATCH_ROWS):
                col.append(UInt32(_cat_code(cat_id, row)))
        cols.append(col^)
    return cols^


def _cat_bin_counts() -> List[Int]:
    var counts = List[Int]()
    for cat_id in range(BATCH_FIRST_CAT_ID + BATCH_CAT_COUNT):
        if cat_id >= BATCH_FIRST_CAT_ID:
            counts.append(_cat_cardinality(cat_id))
        else:
            counts.append(0)
    return counts^


def _expected_ordering(cat_id: Int) raises -> List[UInt32]:
    """What slot `j`'s builder must hold after it has folded `cat_id` into
    the base ordering.

    Independent of the builder: take the base order, STABLE-sort it by the
    feature's code, then mark a segment start wherever the code or the
    BASE segment changes. Two rows sharing a code but sitting in different
    base segments are two segments, which is the whole point of
    `UpdateBordersMask` reading `CurrentBins` and not just `Bins`.
    """
    var order = _base_row_order()
    var starts = _base_segment_starts()

    # base segment id per ORIGINAL row
    var base_seg = List[Int]()
    for _ in range(BATCH_ROWS):
        base_seg.append(-1)
    var seg = -1
    for i in range(len(order)):
        for k in range(len(starts)):
            if starts[k] == i:
                seg += 1
        base_seg[order[i]] = seg

    # stable sort of the base order by code
    var sorted_rows = List[Int]()
    for code in range(_cat_cardinality(cat_id)):
        for i in range(len(order)):
            if _cat_code(cat_id, order[i]) == code:
                sorted_rows.append(order[i])

    var out = List[UInt32]()
    for i in range(len(sorted_rows)):
        var row = sorted_rows[i]
        var flag = i == 0
        if not flag:
            var prev = sorted_rows[i - 1]
            flag = _cat_code(cat_id, row) != _cat_code(cat_id, prev) or (
                base_seg[row] != base_seg[prev]
            )
        out.append(with_mask(UInt32(row), flag))
    return out^


struct RecordingVisitor(TFeatureTensorVisitor):
    """Stands in for `TCtrFromTensorCalcer` (`ctr_from_tensor_calcer.h`),
    which is not ported. It records what the builder handed it, which is
    the only thing this lane is gating."""

    var tensors: List[TFeatureTensor]
    var orderings: List[List[UInt32]]

    def __init__(out self):
        self.tensors = List[TFeatureTensor]()
        self.orderings = List[List[UInt32]]()

    def visit(
        mut self, tensor: TFeatureTensor, mut bin_builder: TCtrBinBuilder
    ) raises:
        self.tensors.append(tensor.copy())
        self.orderings.append(bin_builder.indices.copy())


def _run_batch(width: Int) raises -> RecordingVisitor:
    var builder = TBatchFeatureTensorBuilder(
        _cat_columns(), _cat_bin_counts(), width
    )
    var base = TFeatureTensor()
    base.add_binary_split(TBinarySplit(Int32(1), Int32(5), Int32(1)))
    base.add_binary_split(TBinarySplit(Int32(2), Int32(3), Int32(1)))
    base.add_binary_split(TBinarySplit(Int32(3), Int32(7), Int32(1)))
    var ids = List[UInt32]()
    for k in range(BATCH_CAT_COUNT):
        ids.append(UInt32(BATCH_FIRST_CAT_ID + k))
    var visitor = RecordingVisitor()
    builder.visit_ctr_bin_builders(_base_indices(), base, ids, visitor)
    return visitor^


def main() raises:
    var txt = String(ORACLE)
    var raw = txt.split()
    var toks = List[String]()
    for i in range(len(raw)):
        toks.append(String(raw[i]))
    var pos = 0

    # ---- parse the fixtures ---------------------------------------------
    _expect(toks, pos, "TENSORS")
    var n_tensors = _next_int(toks, pos)
    var fixtures = List[Fixture]()
    for idx in range(n_tensors):
        _expect(toks, pos, "T")
        var got_id = _next_int(toks, pos)
        if got_id != idx:
            raise Error("fixture ids out of order at " + String(idx))
        var f = Fixture()
        var n = _next_int(toks, pos)
        for _ in range(n):
            var a = _next_int(toks, pos)
            var b = _next_int(toks, pos)
            var c = _next_int(toks, pos)
            f.insert_splits.append(
                TBinarySplit(Int32(a), Int32(b), Int32(c))
            )
        n = _next_int(toks, pos)
        for _ in range(n):
            f.insert_cats.append(UInt32(_next_int(toks, pos)))
        n = _next_int(toks, pos)
        for _ in range(n):
            var a = _next_int(toks, pos)
            var b = _next_int(toks, pos)
            var c = _next_int(toks, pos)
            f.canon_splits.append(
                TBinarySplit(Int32(a), Int32(b), Int32(c))
            )
        n = _next_int(toks, pos)
        for _ in range(n):
            f.canon_cats.append(UInt32(_next_int(toks, pos)))
        f.vec_hash = Int32(_next_int(toks, pos))
        f.city = _next_hex(toks, pos)
        f.hash = _next_hex(toks, pos)
        f.size = _next_int(toks, pos)
        f.complexity = _next_int(toks, pos)
        f.simple = _next_int(toks, pos) == 1
        fixtures.append(f^)

    # ---- gate 1: TBinarySplit::GetHash ----------------------------------
    _expect(toks, pos, "SPLITS")
    var n_splits = _next_int(toks, pos)
    var bad_split_hash = 0
    for _ in range(n_splits):
        _expect(toks, pos, "S")
        var a = _next_int(toks, pos)
        var b = _next_int(toks, pos)
        var c = _next_int(toks, pos)
        var want = _next_hex(toks, pos)
        var got = binary_split_hash(TBinarySplit(Int32(a), Int32(b), Int32(c)))
        if got != want:
            bad_split_hash += 1
            print(
                "  MISMATCH split", a, b, c, ":", hex(got), "want", hex(want)
            )
    print("1. split hash:", n_splits, "rows,", bad_split_hash, "wrong")

    # ---- gates 2 and 3 ---------------------------------------------------
    var bad_canon = 0
    var bad_vec = 0
    var bad_city = 0
    var bad_hash = 0
    var bad_scalar = 0
    var bad_order = 0
    var built = List[TFeatureTensor]()
    for idx in range(n_tensors):
        ref f = fixtures[idx]
        var t = _build_forward(f)

        if not _splits_equal(t.get_splits(), f.canon_splits):
            bad_canon += 1
            var msg = String("  MISMATCH canon splits, fixture ") + String(idx)
            for k in range(len(t.get_splits())):
                msg += " " + _split_str(t.get_splits()[k])
            msg += " want"
            for k in range(len(f.canon_splits)):
                msg += " " + _split_str(f.canon_splits[k])
            print(msg)
        if not _cats_equal(t.get_cat_features(), f.canon_cats):
            bad_canon += 1
            print("  MISMATCH canon cats, fixture", idx)

        var vh = t_vec_hash_splits(t.get_splits())
        if vh != f.vec_hash:
            bad_vec += 1
            print(
                "  MISMATCH TVecHash, fixture", idx, ":", vh, "want", f.vec_hash
            )
        var ch = vec_city_hash_u32(t.get_cat_features())
        if ch != f.city:
            bad_city += 1
            print(
                "  MISMATCH VecCityHash, fixture", idx, ":", hex(ch), "want",
                hex(f.city),
            )
        var h = t.get_hash()
        if h != f.hash:
            bad_hash += 1
            print(
                "  MISMATCH GetHash, fixture", idx, ":", hex(h), "want",
                hex(f.hash),
            )
        if (
            t.size() != f.size
            or t.get_complexity() != f.complexity
            or t.is_simple() != f.simple
            or t.is_empty() != (f.size == 0)
        ):
            bad_scalar += 1
            print(
                "  MISMATCH scalars, fixture", idx, ": size", t.size(),
                "complexity", t.get_complexity(), "simple", t.is_simple(),
                "want", f.size, f.complexity, f.simple,
            )

        # gate 3: five more routes to the same value
        var routes = List[TFeatureTensor]()
        routes.append(_build_reverse(f))
        routes.append(_build_interleaved(f))
        routes.append(_build_bulk(f))
        routes.append(_build_by_tensor(f))
        routes.append(_build_doubled(f))
        for r in range(len(routes)):
            if routes[r] != t or routes[r].get_hash() != h:
                bad_order += 1
                print(
                    "  ORDER DEPENDENT, fixture", idx, "route", r, ":",
                    hex(routes[r].get_hash()), "want", hex(h),
                )
        built.append(t^)

    print("2. canonical form:", n_tensors, "fixtures,", bad_canon, "wrong")
    print("2. TVecHash (split half):", bad_vec, "wrong")
    print("2. VecCityHash (cat half):", bad_city, "wrong")
    print("2. GetHash:", bad_hash, "wrong")
    print("2. Size/Complexity/IsSimple/IsEmpty:", bad_scalar, "wrong")
    print(
        "3. order independence:", n_tensors * 5, "rebuilds,", bad_order,
        "disagreed",
    )

    # ---- gate 4: pairs ---------------------------------------------------
    _expect(toks, pos, "PAIRS")
    var n_pairs = _next_int(toks, pos)
    if n_pairs != n_tensors:
        raise Error("PAIRS header disagrees with TENSORS")
    var bad_eq = 0
    var bad_lt = 0
    var bad_sub = 0
    var bad_property = 0
    for i in range(n_pairs):
        _expect(toks, pos, "P")
        var row = _next_int(toks, pos)
        if row != i:
            raise Error("pair rows out of order")
        var masks = List[String]()
        masks.append(_next_str(toks, pos))
        masks.append(_next_str(toks, pos))
        masks.append(_next_str(toks, pos))
        for j in range(n_tensors):
            var nib_eq = _hex_digit(masks[0].as_bytes()[j // 4])
            var nib_lt = _hex_digit(masks[1].as_bytes()[j // 4])
            var nib_sub = _hex_digit(masks[2].as_bytes()[j // 4])
            var want_eq = ((nib_eq >> (j % 4)) & 1) == 1
            var want_lt = ((nib_lt >> (j % 4)) & 1) == 1
            var want_sub = ((nib_sub >> (j % 4)) & 1) == 1
            var got_eq = built[i] == built[j]
            var got_lt = built[i].less(built[j])
            var got_sub = built[i].is_subset(built[j])
            if got_eq != want_eq:
                bad_eq += 1
                print("  MISMATCH ==", i, j, got_eq, "want", want_eq)
            if got_lt != want_lt:
                bad_lt += 1
                print("  MISMATCH <", i, j, got_lt, "want", want_lt)
            if got_sub != want_sub:
                bad_sub += 1
                print("  MISMATCH IsSubset", i, j, got_sub, "want", want_sub)
            # properties the oracle does not have to be trusted for
            var back_lt = built[j].less(built[i])
            var trichotomy = 0
            if got_lt:
                trichotomy += 1
            if back_lt:
                trichotomy += 1
            if got_eq:
                trichotomy += 1
            if trichotomy != 1:
                bad_property += 1
                print("  NOT A TOTAL ORDER at", i, j)
            if got_eq and built[i].get_hash() != built[j].get_hash():
                bad_property += 1
                print("  EQUAL BUT HASHES DIFFER at", i, j)
            var got_ne = built[i] != built[j]
            if got_eq == got_ne:
                bad_property += 1
                print("  __ne__ disagrees with __eq__ at", i, j)
    print("4. ==:", bad_eq, "wrong; <:", bad_lt, "wrong; IsSubset:", bad_sub,
          "wrong; properties:", bad_property, "violations")

    # ---- gate 5: AddTensor ----------------------------------------------
    _expect(toks, pos, "MERGES")
    var n_merges = _next_int(toks, pos)
    var bad_merge = 0
    for _ in range(n_merges):
        _expect(toks, pos, "M")
        var i = _next_int(toks, pos)
        var j = _next_int(toks, pos)
        var want_hash = _next_hex(toks, pos)
        var want_size = _next_int(toks, pos)
        var want_cpx = _next_int(toks, pos)
        var m = built[i].copy()
        m.add_tensor(built[j])
        if (
            m.get_hash() != want_hash
            or m.size() != want_size
            or m.get_complexity() != want_cpx
        ):
            bad_merge += 1
            print(
                "  MISMATCH AddTensor", i, j, ":", hex(m.get_hash()), m.size(),
                m.get_complexity(), "want", hex(want_hash), want_size,
                want_cpx,
            )
        # a merge is a UNION, so both operands are subsets of it
        if not built[i].is_subset(m) or not built[j].is_subset(m):
            bad_merge += 1
            print("  AddTensor is not a union at", i, j)
    print("5. AddTensor:", n_merges, "rows,", bad_merge, "wrong")

    # ---- gate 6: collisions ---------------------------------------------
    _expect(toks, pos, "ENUM")
    var want_count = _next_int(toks, pos)
    var want_distinct = _next_int(toks, pos)
    var want_xor = _next_hex(toks, pos)
    var want_sum = _next_hex(toks, pos)

    var hashes = List[UInt64]()
    for f in range(6):
        for b in range(6):
            for ty in range(2):
                for c in range(10):
                    var x = TFeatureTensor()
                    x.add_binary_split(
                        TBinarySplit(Int32(f), Int32(b), Int32(ty))
                    )
                    x.add_cat_feature(UInt32(1000 + c))
                    hashes.append(x.get_hash())
    for f in range(4):
        for b in range(4):
            for c in range(5):
                var x = TFeatureTensor()
                x.add_binary_split(TBinarySplit(Int32(f), Int32(b), Int32(0)))
                x.add_binary_split(
                    TBinarySplit(Int32(f + 10), Int32(b + 1), Int32(1))
                )
                x.add_cat_feature(UInt32(c))
                x.add_cat_feature(UInt32(c + 1))
                hashes.append(x.get_hash())

    var got_xor = UInt64(0)
    var got_sum = UInt64(0)
    for i in range(len(hashes)):
        got_xor ^= hashes[i]
        got_sum += hashes[i]
    var collisions = 0
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hashes[i] == hashes[j]:
                collisions += 1
    var bad_enum = 0
    if len(hashes) != want_count:
        bad_enum += 1
        print("  enumerated", len(hashes), "want", want_count)
    if collisions != 0:
        bad_enum += 1
        print("  COLLISIONS:", collisions, "pairs of DIFFERENT tensors hash equal")
    if len(hashes) - collisions != want_distinct:
        bad_enum += 1
    if got_xor != want_xor:
        bad_enum += 1
        print("  xor", hex(got_xor), "want", hex(want_xor))
    if got_sum != want_sum:
        bad_enum += 1
        print("  sum", hex(got_sum), "want", hex(want_sum))
    print(
        "6. collisions:", len(hashes), "distinct tensors,", collisions,
        "hash collisions,", bad_enum, "disagreements with the oracle",
    )

    # ---- gate 7: the two feature-manager predicates ----------------------
    var bad_pred = 0
    if is_tree_ctrs_enabled(False, 4):
        bad_pred += 1
        print("  IsTreeCtrsEnabled true with no categorical features")
    if is_tree_ctrs_enabled(True, 1):
        bad_pred += 1
        print("  IsTreeCtrsEnabled true at MaxTensorComplexity 1 (our pin)")
    if not is_tree_ctrs_enabled(True, 2):
        bad_pred += 1
        print("  IsTreeCtrsEnabled false at MaxTensorComplexity 2")
    if not is_tree_ctrs_enabled(True, 4):
        bad_pred += 1
        print("  IsTreeCtrsEnabled false at their default 4")
    # complexity 4 (fixture 49: five splits + three cats) is NOT < 4
    if use_as_base_tensor_for_tree_ctr(built[49], 4):
        bad_pred += 1
        print("  UseAsBaseTensor accepted complexity 4 at limit 4")
    if not use_as_base_tensor_for_tree_ctr(built[49], 5):
        bad_pred += 1
        print("  UseAsBaseTensor refused complexity 4 at limit 5")
    # fixture 50 is the same five splits with two cats: complexity 3
    if not use_as_base_tensor_for_tree_ctr(built[50], 4):
        bad_pred += 1
        print("  UseAsBaseTensor refused complexity 3 at limit 4")
    # fixture 45 is three splits and no cats: complexity ONE, not three
    if built[45].get_complexity() != 1 or built[45].size() != 3:
        bad_pred += 1
        print("  GetComplexity read as Size for a splits-only tensor")
    if use_as_base_tensor_for_tree_ctr(built[45], 1):
        bad_pred += 1
        print("  UseAsBaseTensor accepted complexity 1 at limit 1")
    print("7. manager predicates:", bad_pred, "wrong")

    # ---- gate 8: the batch builder --------------------------------------
    var bad_batch = 0
    var v3 = _run_batch(3)
    if len(v3.tensors) != BATCH_CAT_COUNT:
        bad_batch += 1
        print("  visited", len(v3.tensors), "tensors, want", BATCH_CAT_COUNT)
    for k in range(len(v3.tensors)):
        var cat_id = BATCH_FIRST_CAT_ID + k
        var want_tensor = built[45].copy()
        want_tensor.add_cat_feature(UInt32(cat_id))
        if v3.tensors[k] != want_tensor:
            bad_batch += 1
            print("  visit", k, "carried the wrong tensor")
        var want_order = _expected_ordering(cat_id)
        if len(v3.orderings[k]) != len(want_order):
            bad_batch += 1
            print("  visit", k, "ordering has the wrong length")
        else:
            for i in range(len(want_order)):
                if v3.orderings[k][i] != want_order[i]:
                    bad_batch += 1
                    print(
                        "  visit", k, "position", i, ": row",
                        index_of(v3.orderings[k][i]), "flag",
                        is_segment_start(v3.orderings[k][i]), "want row",
                        index_of(want_order[i]), "flag",
                        is_segment_start(want_order[i]),
                    )
                    break
    # the first three tensors are oracle fixtures 46, 47 and 48
    var oracle_ids: List[Int] = [46, 47, 48]
    for k in range(3):
        if v3.tensors[k].get_hash() != fixtures[oracle_ids[k]].hash:
            bad_batch += 1
            print(
                "  visit", k, "hash", hex(v3.tensors[k].get_hash()), "want",
                hex(fixtures[oracle_ids[k]].hash),
            )
    # the batch width must change nothing that comes out
    for w in range(4):
        var width: List[Int] = [1, 2, 7, 100]
        var vw = _run_batch(width[w])
        if len(vw.tensors) != len(v3.tensors):
            bad_batch += 1
            print("  width", width[w], "visited a different number of tensors")
            continue
        for k in range(len(vw.tensors)):
            if vw.tensors[k] != v3.tensors[k]:
                bad_batch += 1
                print("  width", width[w], "visit", k, "tensor differs")
            if not _cats_equal(vw.orderings[k], v3.orderings[k]):
                bad_batch += 1
                print("  width", width[w], "visit", k, "ordering differs")
    # RequestStream grows the pool and never shrinks it
    var pool = TBatchFeatureTensorBuilder(
        _cat_columns(), _cat_bin_counts(), 3
    )
    if pool.request_stream(2) != 2 or len(pool.ctr_bin_builders) != 2:
        bad_batch += 1
        print("  RequestStream(2) did not open exactly two slots")
    if pool.request_stream(5) != 3 or len(pool.ctr_bin_builders) != 3:
        bad_batch += 1
        print("  RequestStream(5) did not clamp to TensorBuilderStreams")
    if pool.request_stream(1) != 1 or len(pool.ctr_bin_builders) != 3:
        bad_batch += 1
        print("  RequestStream(1) shrank the pool")
    print(
        "8. batch builder:", len(v3.tensors), "visits at width 3 in",
        (BATCH_CAT_COUNT + 2) // 3, "groups, plus widths 1/2/7/100;",
        bad_batch, "wrong",
    )

    var total = (
        bad_split_hash + bad_canon + bad_vec + bad_city + bad_hash
        + bad_scalar + bad_order + bad_eq + bad_lt + bad_sub + bad_property
        + bad_merge + bad_enum + bad_pred + bad_batch
    )
    if total != 0:
        raise Error("feature tensor check FAILED: " + String(total) + " failures")
    print("feature tensor check OK")
