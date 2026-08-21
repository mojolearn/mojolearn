// Oracle for gbdt/methods/batch_feature_tensor_builder.mojo.
//
// Everything above main() is a TRANSCRIPTION of CatBoost's own source at
// 54a8143a, cited by file, compiled by clang. `city.h`/`city.cpp` are taken
// from tools/cityhash_oracle/, which are their util/digest/city.{h,cpp} byte
// for byte.
//
//   util/digest/numeric.h:41-63   IntHashImpl(ui32), IntHashImpl(ui64)
//   util/digest/numeric.h:65-70   IntHash<T>
//   util/digest/numeric.h:75-88   NumericHash<T>
//   util/digest/numeric.h:90-93   CombineHashes<T>
//   util/digest/multi.h:6-14      MultiHash
//   util/str_stl.h:37-53,159-160  THash<T> -> hash<T> -> THashHelper
//   catboost/libs/model/hash.h:16-25       TVecHash<T>
//   catboost/libs/helpers/hash.h:6-9       VecCityHash
//   catboost/libs/helpers/set.h:7-10       NCB::IsSubset
//   catboost/cuda/data/feature.h:35-71     TBinarySplit
//   catboost/cuda/data/feature.h:89-186    TFeatureTensor
#include "city.h"

#include <algorithm>
#include <cstdio>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <vector>

// ---- util/digest/numeric.h ---------------------------------------------
static constexpr ui32 IntHashImpl(ui32 key) noexcept {
    key += ~(key << 15);
    key ^= (key >> 10);
    key += (key << 3);
    key ^= (key >> 6);
    key += ~(key << 11);
    key ^= (key >> 16);
    return key;
}

static constexpr ui64 IntHashImpl(ui64 key) noexcept {
    key += ~(key << 32);
    key ^= (key >> 22);
    key += ~(key << 13);
    key ^= (key >> 8);
    key += (key << 3);
    key ^= (key >> 15);
    key += ~(key << 27);
    key ^= (key >> 31);
    return key;
}

// numeric.h:65-70: IntHash<T> converts to the SAME-WIDTH unsigned first
// (TFixedWidthUnsignedInt<T>), which is what picks the ui64 mixer for size_t.
template <class T>
static constexpr T IntHash(T t) noexcept {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "no fixed-width sibling");
    if constexpr (sizeof(T) == 8) {
        return (T)IntHashImpl((ui64)t);
    } else {
        return (T)IntHashImpl((ui32)t);
    }
}

// numeric.h:90-93
template <class T>
static constexpr T CombineHashes(T l, T r) noexcept {
    return IntHash(l) ^ r;
}

// ---- catboost/cuda/data/feature.h:21-24 --------------------------------
enum class EBinSplitType { TakeBin, TakeGreater };

// str_stl.h: THash<enum> is NumericHash, THash<integral> is the value cast.
static size_t HashOfSplitType(EBinSplitType t) {
    // NumericHash: union-punned to the same-width unsigned, then IntHash.
    static_assert(sizeof(EBinSplitType) == 4, "underlying type must be 4 bytes");
    union { EBinSplitType t; ui32 cvt; } u{t};
    return (size_t)IntHashImpl(u.cvt);
}

// ---- catboost/cuda/data/feature.h:35-71 --------------------------------
struct TBinarySplit {
    ui32 FeatureId = 0;
    ui32 BinIdx = 0;
    EBinSplitType SplitType = EBinSplitType::TakeBin;

    TBinarySplit(ui32 f, ui32 b, EBinSplitType t)
        : FeatureId(f), BinIdx(b), SplitType(t) {}
    TBinarySplit() = default;

    bool operator<(const TBinarySplit& o) const {
        return std::tie(FeatureId, BinIdx, SplitType) <
               std::tie(o.FeatureId, o.BinIdx, o.SplitType);
    }
    bool operator==(const TBinarySplit& o) const {
        return std::tie(FeatureId, BinIdx, SplitType) ==
               std::tie(o.FeatureId, o.BinIdx, o.SplitType);
    }
    // MultiHash(FeatureId, BinIdx, SplitType), multi.h:6-14
    ui64 GetHash() const {
        size_t h = HashOfSplitType(SplitType);
        h = CombineHashes<size_t>(h, (size_t)BinIdx);
        h = CombineHashes<size_t>(h, (size_t)FeatureId);
        return h;
    }
};

// ---- catboost/libs/model/hash.h:16-25 ----------------------------------
static int TVecHashSplits(const std::vector<TBinarySplit>& a) {
    ui32 res = 1988712;
    for (size_t i = 0; i < a.size(); ++i) {
        res = 984121 * res + (ui32)a[i].GetHash();  // ui32 truncation is theirs
    }
    return static_cast<int>(res);
}

// ---- catboost/libs/helpers/hash.h:6-9 ----------------------------------
static ui64 VecCityHash(const std::vector<ui32>& data) {
    return CityHash64(reinterpret_cast<const char*>(data.data()),
                      sizeof(ui32) * data.size());
}

// ---- catboost/cuda/data/feature.h:89-186 -------------------------------
template <class V>
static void Unique(V& v) {
    ui64 size = std::unique(v.begin(), v.end()) - v.begin();
    v.resize(size);
}

struct TFeatureTensor {
    std::vector<TBinarySplit> Splits;
    std::vector<ui32> CatFeatures;

    bool IsSimple() const { return (Splits.size() + CatFeatures.size()) == 1; }

    void SortUniqueSplits() {
        std::sort(Splits.begin(), Splits.end());
        Unique(Splits);
    }
    void SortUniqueCatFeatures() {
        std::sort(CatFeatures.begin(), CatFeatures.end());
        Unique(CatFeatures);
    }
    TFeatureTensor& AddBinarySplit(const TBinarySplit& bin) {
        Splits.push_back(bin);
        SortUniqueSplits();
        return *this;
    }
    TFeatureTensor& AddCatFeature(ui32 featureId) {
        CatFeatures.push_back(featureId);
        SortUniqueCatFeatures();
        return *this;
    }
    TFeatureTensor& AddTensor(const TFeatureTensor& t) {
        for (auto& s : t.Splits) Splits.push_back(s);
        for (auto& c : t.CatFeatures) CatFeatures.push_back(c);
        SortUniqueSplits();
        SortUniqueCatFeatures();
        return *this;
    }
    bool operator==(const TFeatureTensor& o) const {
        return Splits == o.Splits && CatFeatures == o.CatFeatures;
    }
    bool operator<(const TFeatureTensor& o) const {
        return std::tie(Splits, CatFeatures) < std::tie(o.Splits, o.CatFeatures);
    }
    bool IsEmpty() const { return CatFeatures.empty() && Splits.empty(); }
    ui64 Size() const { return CatFeatures.size() + Splits.size(); }
    ui64 GetHash() const {
        // MultiHash(TVecHash<TBinarySplit>()(Splits), VecCityHash(CatFeatures))
        size_t h = (size_t)VecCityHash(CatFeatures);
        h = CombineHashes<size_t>(h, (size_t)TVecHashSplits(Splits));
        return h;
    }
    bool IsSubset(const TFeatureTensor& o) const {
        return std::includes(o.Splits.begin(), o.Splits.end(),
                             Splits.begin(), Splits.end()) &&
               std::includes(o.CatFeatures.begin(), o.CatFeatures.end(),
                             CatFeatures.begin(), CatFeatures.end());
    }
    ui64 GetComplexity() const {
        return CatFeatures.size() + std::min<ui64>(Splits.size(), 1);
    }
};

// ---- the fixtures ------------------------------------------------------
struct TSpec {
    std::vector<TBinarySplit> SplitInserts;
    std::vector<ui32> CatInserts;
};

static TBinarySplit S(ui32 f, ui32 b, ui32 t) {
    return TBinarySplit(f, b, t == 0 ? EBinSplitType::TakeBin
                                     : EBinSplitType::TakeGreater);
}

int main() {
    std::vector<TSpec> specs;

    auto add = [&](std::vector<TBinarySplit> ss, std::vector<ui32> cs) {
        specs.push_back(TSpec{ss, cs});
    };

    // 0 empty
    add({}, {});
    // 1 one split only
    add({S(3, 1, 0)}, {});
    // 2 one cat only (the SIMPLE ctr tensor)
    add({}, {7});
    // 3..6 one split + one cat, each field perturbed one at a time
    add({S(3, 1, 0)}, {7});
    add({S(3, 1, 1)}, {7});   // split type only
    add({S(3, 2, 0)}, {7});   // bin only
    add({S(4, 1, 0)}, {7});   // feature only
    // 7 cat only perturbed
    add({S(3, 1, 0)}, {8});
    // 8 two splits + two cats
    add({S(3, 1, 0), S(4, 2, 1)}, {7, 8});
    // 9 SAME tensor, every insertion reversed
    add({S(4, 2, 1), S(3, 1, 0)}, {8, 7});
    // 10 SAME tensor, duplicates inserted, interleaved order
    add({S(4, 2, 1), S(3, 1, 0), S(4, 2, 1), S(3, 1, 0)}, {8, 7, 7, 8, 8});
    // 11 SAME split multiset, cats swapped for the same two in the other order
    add({S(3, 1, 0), S(4, 2, 1)}, {8, 7});
    // 12..31 cat-vector byte-length sweep: 0,4,8,...,80 bytes plus the
    // >64-byte CityHash loop at 1 and 2 iterations.
    for (int n = 1; n <= 20; ++n) {
        std::vector<ui32> cs;
        for (int i = 0; i < n; ++i) cs.push_back((ui32)(1000 + 7 * i));
        add({S(3, 1, 0)}, cs);
    }
    // 32 33 cats -> 132 bytes -> two loop iterations
    {
        std::vector<ui32> cs;
        for (int i = 0; i < 33; ++i) cs.push_back((ui32)(2000 + 3 * i));
        add({S(3, 1, 0)}, cs);
    }
    // 33..40 split-count sweep 1..8 (the TVecHash fold length)
    for (int n = 1; n <= 8; ++n) {
        std::vector<TBinarySplit> ss;
        for (int i = 0; i < n; ++i) ss.push_back(S(10 + i, 2 * i + 1, i % 2));
        add(ss, {5});
    }
    // 41 ui32 values at and above 2^31 in every field: theirs are ui32 and
    // both the comparator and THash must read them unsigned.
    add({S(0x80000001u, 0xFFFFFFFFu, 1), S(0x7FFFFFFFu, 0u, 0)},
        {0x80000000u, 0xFFFFFFFFu, 1u});
    // 42 the same two splits inserted the other way round
    add({S(0x7FFFFFFFu, 0u, 0), S(0x80000001u, 0xFFFFFFFFu, 1)},
        {0xFFFFFFFFu, 1u, 0x80000000u});
    // 43 splits sorted only by the THIRD key
    add({S(9, 9, 1), S(9, 9, 0)}, {});
    // 44 splits sorted only by the SECOND key
    add({S(9, 9, 0), S(9, 8, 0)}, {});
    // 45 base tensor of depth 3 with no cats (a tree-ctr BASE)
    add({S(1, 5, 1), S(2, 3, 1), S(3, 7, 1)}, {});
    // 46..48 that base plus one cat each -- what the batch builder emits
    add({S(1, 5, 1), S(2, 3, 1), S(3, 7, 1)}, {11});
    add({S(1, 5, 1), S(2, 3, 1), S(3, 7, 1)}, {12});
    add({S(1, 5, 1), S(2, 3, 1), S(3, 7, 1)}, {13});
    // 49 complexity boundary: 5 splits + 3 cats -> complexity 4
    add({S(1, 1, 1), S(2, 1, 1), S(3, 1, 1), S(4, 1, 1), S(5, 1, 1)},
        {20, 21, 22});
    // 50 5 splits + 2 cats -> complexity 3
    add({S(1, 1, 1), S(2, 1, 1), S(3, 1, 1), S(4, 1, 1), S(5, 1, 1)},
        {20, 21});
    // 51 0 splits + 4 cats -> complexity 4
    add({}, {20, 21, 22, 23});

    printf("TENSORS %d\n", (int)specs.size());
    std::vector<TFeatureTensor> built;
    for (size_t i = 0; i < specs.size(); ++i) {
        TFeatureTensor t;
        for (auto& s : specs[i].SplitInserts) t.AddBinarySplit(s);
        for (auto c : specs[i].CatInserts) t.AddCatFeature(c);
        built.push_back(t);

        printf("T %d", (int)i);
        printf(" %d", (int)specs[i].SplitInserts.size());
        for (auto& s : specs[i].SplitInserts)
            printf(" %u %u %u", s.FeatureId, s.BinIdx, (ui32)s.SplitType);
        printf(" %d", (int)specs[i].CatInserts.size());
        for (auto c : specs[i].CatInserts) printf(" %u", c);
        printf(" %d", (int)t.Splits.size());
        for (auto& s : t.Splits)
            printf(" %u %u %u", s.FeatureId, s.BinIdx, (ui32)s.SplitType);
        printf(" %d", (int)t.CatFeatures.size());
        for (auto c : t.CatFeatures) printf(" %u", c);
        printf(" %d %016llx %016llx %d %d %d\n",
               TVecHashSplits(t.Splits),
               (unsigned long long)VecCityHash(t.CatFeatures),
               (unsigned long long)t.GetHash(),
               (int)t.Size(), (int)t.GetComplexity(), (int)t.IsSimple());
    }

    // Every split that appears anywhere, with its own GetHash.
    std::vector<TBinarySplit> allSplits;
    for (auto& sp : specs)
        for (auto& s : sp.SplitInserts) allSplits.push_back(s);
    std::sort(allSplits.begin(), allSplits.end());
    Unique(allSplits);
    printf("SPLITS %d\n", (int)allSplits.size());
    for (auto& s : allSplits)
        printf("S %u %u %u %016llx\n", s.FeatureId, s.BinIdx, (ui32)s.SplitType,
               (unsigned long long)s.GetHash());

    // Every ordered pair, one row per i: three bitmasks over j, low bit
    // first, printed as hex nibble groups of 4 j-values.
    printf("PAIRS %d\n", (int)built.size());
    for (size_t i = 0; i < built.size(); ++i) {
        printf("P %d ", (int)i);
        const char* names[3] = {"", "", ""};
        (void)names;
        for (int which = 0; which < 3; ++which) {
            for (size_t base = 0; base < built.size(); base += 4) {
                int nib = 0;
                for (size_t k = 0; k < 4 && base + k < built.size(); ++k) {
                    size_t j = base + k;
                    int bit = which == 0 ? (int)(built[i] == built[j])
                            : which == 1 ? (int)(built[i] < built[j])
                                         : (int)built[i].IsSubset(built[j]);
                    nib |= bit << k;
                }
                printf("%x", nib);
            }
            printf(" ");
        }
        printf("\n");
    }

    // AddTensor: tensor i merged with tensor j, for a diagonal band.
    printf("MERGES %d\n", (int)built.size());
    for (size_t i = 0; i < built.size(); ++i) {
        size_t j = (i * 7 + 3) % built.size();
        TFeatureTensor m = built[i];
        m.AddTensor(built[j]);
        printf("M %d %d %016llx %d %d\n", (int)i, (int)j,
               (unsigned long long)m.GetHash(), (int)m.Size(),
               (int)m.GetComplexity());
    }
    // The COLLISION set: 800 tensors that are pairwise DIFFERENT by
    // construction. A hash that returns a constant, or one that ignores a
    // field, collapses Distinct.
    {
        std::vector<ui64> hashes;
        for (ui32 f = 0; f < 6; ++f)
            for (ui32 b = 0; b < 6; ++b)
                for (ui32 t = 0; t < 2; ++t)
                    for (ui32 c = 0; c < 10; ++c) {
                        TFeatureTensor x;
                        x.AddBinarySplit(S(f, b, t));
                        x.AddCatFeature(1000 + c);
                        hashes.push_back(x.GetHash());
                    }
        for (ui32 f = 0; f < 4; ++f)
            for (ui32 b = 0; b < 4; ++b)
                for (ui32 c = 0; c < 5; ++c) {
                    TFeatureTensor x;
                    x.AddBinarySplit(S(f, b, 0));
                    x.AddBinarySplit(S(f + 10, b + 1, 1));
                    x.AddCatFeature(c);
                    x.AddCatFeature(c + 1);
                    hashes.push_back(x.GetHash());
                }
        ui64 x = 0, sum = 0;
        for (ui64 h : hashes) { x ^= h; sum += h; }
        std::vector<ui64> sorted = hashes;
        std::sort(sorted.begin(), sorted.end());
        size_t distinct = std::unique(sorted.begin(), sorted.end()) - sorted.begin();
        printf("ENUM %d %d %016llx %016llx\n", (int)hashes.size(), (int)distinct,
               (unsigned long long)x, (unsigned long long)sum);
    }
    return 0;
}
