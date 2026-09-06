# The category hash oracle

Generates `bench/cityhash_oracle.txt`, which `pixi run check-cityhash`
reads. Regenerate with

    c++ -std=c++17 -O2 -I tools/cityhash_oracle \
        tools/cityhash_oracle/oracle.cpp \
        tools/cityhash_oracle/city.cpp -o /tmp/cityhash_oracle
    /tmp/cityhash_oracle > bench/cityhash_oracle.txt

`city.h`, `city.cpp` and `city_streaming.h` are CatBoost's own files at
`54a8143a` (`util/digest/`), byte for byte apart from the `util/` includes
that `shim.h` replaces with the standard-library pieces they resolve to.
So the hash IS their hash, not a description of it.

That matters more here than usual: their `city.h` states outright that it
is a **CityHash 1.0 implementation whose results are different from the
mainline version of CityHash**. Google's published test vectors, or a port
written from Google's current sources, would therefore be checked against
the wrong function and agree on nothing. There is no third-party oracle;
their file is the only ground truth.

`oracle.cpp` transcribes the two one-liners above the hash --
`CalcCatFeatureHash` (`catboost/libs/cat_feature/cat_feature.cpp:6-8`) and
`CalcHash` (`catboost/libs/model/hash.h:11-14`) -- and emits the
`(ui64)(int)` sign-extension of `CalcHashes`
(`catboost/libs/model/ctr_provider.h:107`) in its chain rows. That is a
second implementation in a different language through a different
compiler, which is what makes the comparison evidence rather than a
tautology. The chain rows earned their keep on the first run: they caught
the Mojo side zero-extending a category hash where their cast
sign-extends, a defect visible only for hashes at or above 2^31.
