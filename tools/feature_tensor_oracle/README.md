# `TFeatureTensor` oracle

CatBoost's own source text, compiled by clang, printing the canonical form,
hash, ordering and subset relation of 52 tensor fixtures plus an 800-tensor
collision sweep. Its output is embedded verbatim in
`checks/feature_tensor_check.mojo`, so the check is self-contained and
this directory only has to be rebuilt when a fixture changes.

    c++ -std=c++17 -O2 -I ../cityhash_oracle \
        oracle.cpp ../cityhash_oracle/city.cpp -o oracle && ./oracle

What it transcribes, all read out of `/private/tmp/catboost-src` at
`54a8143a`: `util/digest/numeric.h`, `util/digest/multi.h`,
`util/generic/str_stl.h`, `catboost/libs/model/hash.h`,
`catboost/cuda/cuda_util/helpers.h`, `library/cpp/containers/2d_array` set
helpers, and the `TFeatureTensor` half of `catboost/cuda/data/feature.h`.
CityHash comes from `tools/cityhash_oracle/city.cpp`, which is CatBoost's
vendored CityHash 1.0 and is already the oracle for `pixi run check-cityhash`.

Each fixture's INSERTION ORDER is read out of this file by the Mojo check
rather than restated there, so the two sides cannot drift about what they
built -- which matters precisely because their canonicalisation erases
insertion order (DEVIATION 116b) and a check that built a different tensor
would still agree.
