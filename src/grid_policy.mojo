"""Feature grouping and bit packing.

PORT OF `catboost/cuda/gpu_data/grid_policy.h` at CatBoost `54a8143a`.
Transliterated. Do not improve.

This is the file that makes CatBoost's histogram read 32 features out of one
4-byte load on binary data. A feature is assigned to one of three policies by
how many bins it needs, and the policy fixes how many features share a
`UInt32` of the compressed index:

    BinaryFeatures    1 bit  per feature  -> 32 features per UInt32
    HalfByteFeatures  4 bits per feature  ->  8 features per UInt32
    OneByteFeatures   8 bits per feature  ->  4 features per UInt32

The histogram kernel's group size matches the packing, so the kernel loads
one `UInt32` and extracts every feature in it by shift and mask.

Their layout detail worth not "fixing": `Shift` counts from the TOP of the
word, `32 - (1 + localId) * BitsPerFeature`, so feature 0 of a group lives in
the HIGH bits. It is arbitrary and it must be copied exactly, because the
kernels index against it.
"""


# The three policies. Mojo has no scoped enum, so these are the codes.
comptime POLICY_BINARY = 0
comptime POLICY_HALF_BYTE = 1
comptime POLICY_ONE_BYTE = 2


def bits_per_feature(policy: Int) -> Int:
    """`TFeaturePolicyTraits<Policy>::BitsPerFeature()`."""
    if policy == POLICY_BINARY:
        return 1
    if policy == POLICY_HALF_BYTE:
        return 4
    return 8


def features_per_int(policy: Int) -> Int:
    """`TFeaturePolicyTraits<Policy>::FeaturesPerInt()`, i.e. 32 / bits."""
    return 32 // bits_per_feature(policy)


def policy_mask(policy: Int) -> UInt32:
    """`TCompressedIndexHelper::Mask()`, `(1 << bits) - 1`."""
    return (UInt32(1) << UInt32(bits_per_feature(policy))) - 1


def policy_bin_count(policy: Int) -> Int:
    """`TCompressedIndexHelper::BinCount()`, `1 << bits`.

    The number of bin values the packing can REPRESENT, which is one more
    than the number of folds a feature may use.
    """
    return 1 << bits_per_feature(policy)


def policy_max_folds(policy: Int) -> Int:
    """`TCompressedIndexHelper::MaxFolds()`, `(1 << bits) - 1`.

    A feature needing more folds than this cannot live under the policy. The
    reserved value is what makes a binary feature 1 bit rather than 2.
    """
    return (1 << bits_per_feature(policy)) - 1


def policy_shift(policy: Int, feature_id: Int) -> Int:
    """`TCompressedIndexHelper::Shift(featureId)`.

    COPIED EXACTLY, high bits first:
        `32 - (1 + localId) * BitsPerFeature()`
    """
    var local_id = feature_id % features_per_int(policy)
    return 32 - (1 + local_id) * bits_per_feature(policy)


def policy_shifted_mask(policy: Int, feature_id: Int) -> UInt32:
    """`TCompressedIndexHelper::ShiftedMask(featureId)`."""
    return policy_mask(policy) << UInt32(policy_shift(policy, feature_id))


def policy_for_fold_count(fold_count: Int) -> Int:
    """The policy a feature with `fold_count` folds is placed under.

    CatBoost assigns this while building the compressed index rather than in
    `grid_policy.h`; the rule is the smallest policy whose `MaxFolds` covers
    the feature, which is what makes covtype's 0/1 columns binary and
    therefore 32-to-a-word.
    """
    if fold_count <= policy_max_folds(POLICY_BINARY):
        return POLICY_BINARY
    if fold_count <= policy_max_folds(POLICY_HALF_BYTE):
        return POLICY_HALF_BYTE
    return POLICY_ONE_BYTE


def policy_name(policy: Int) -> String:
    if policy == POLICY_BINARY:
        return String("BinaryFeatures")
    if policy == POLICY_HALF_BYTE:
        return String("HalfByteFeatures")
    return String("OneByteFeatures")
