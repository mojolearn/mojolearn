# The CTR estimation permutation oracle

Generates `bench/permutation_oracle.txt`, which `pixi run check-permutation`
reads. Regenerate with

    c++ -std=c++17 -O2 -I tools/permutation_oracle \
        tools/permutation_oracle/oracle.cpp \
        tools/permutation_oracle/mersenne64.cpp -o /tmp/permutation_oracle
    /tmp/permutation_oracle > bench/permutation_oracle.txt

`mersenne64.h` and `mersenne64.cpp` are CatBoost's own files at `54a8143a`
(`util/random/mersenne64.{h,cpp}`), byte for byte apart from two `util/`
includes that `shim.h` replaces and the `IInputStream` constructor, which
needs their stream library and which `TRandom` never calls. So the generator
IS their generator, not a description of it.

`oracle.cpp` transcribes the four short pieces around it -- `GenUniform`
(`util/random/common_ops.h:48-60`), `TRandom`
(`catboost/libs/helpers/cpu_random.h:17-33`), `Shuffle`
(`catboost/cuda/data/data_utils.h:21-47`) and `GetSeed`
(`catboost/cuda/data/permutation.h:93-95`) -- in C++, compiled by clang. That
is a second implementation in a different language from the Mojo one, which is
what makes the comparison evidence rather than a tautology.

There is no live CatBoost oracle for this: their GPU learner refuses to run on
Apple silicon, and their CPU learner never exposes a CTR estimation order.
