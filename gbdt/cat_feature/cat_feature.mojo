"""`CalcCatFeatureHash`: what a raw category IS to CatBoost.

PORT OF `catboost/libs/cat_feature/cat_feature.cpp:6-8` at `54a8143a`: the
low 32 bits of their own CityHash64 1.0 variant (`gbdt/digest/city.mojo`)
over the category's raw bytes. This single ui32 is the identity of a
category everywhere in their system: the pool, the training bins' perfect
hash, the model file's `ctr_data.hash_map` key, and the first element of
the apply-time combination chain (`libs/model/ctr_provider.h:107`).

This port trains on dense sorted-unique codes instead (see
`tools/ctr_prep.py`'s note: a CTR value depends only on counts, so the code
LABELS cannot change any statistic). The hash is therefore not on the
training path here and stays unwired until one of its two real callers
lands: scoring raw strings without a prep script, or reading/writing a
model file whose CTR tables must interop with theirs.

The float bitcasts beside it in their header (`ConvertCatFeatureHashToFloat`
/ `ConvertFloatCatFeatureToIntHash`, `cat_feature.h:14-20`) are NOT PORTED
YET: they exist because their pools store cat features as bit-punned
float32, a storage decision this port has not adopted.

Gated by `pixi run check-cityhash` against their own file compiled by
`tools/cityhash_oracle/`.
"""

from gbdt.digest.city import city_hash_64


def calc_cat_feature_hash(s: Span[UInt8, _]) -> UInt32:
    # cat_feature.cpp:6-8: CityHash64(feature) & 0xffffffff.
    return (city_hash_64(s) & 0xFFFFFFFF).cast[DType.uint32]()
