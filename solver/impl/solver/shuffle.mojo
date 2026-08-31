# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`cuml/cpp/src/solver/shuffle.h` -- `initShuffle` and `shuffle`.

    initShuffle(rand_indices, g, random_state = 0):      shuffle.h:15-20
        g.seed((int)random_state)
        rand_indices[i] = i  for every i

    shuffle(rand_indices, g):                            shuffle.h:22-26
        std::shuffle(rand_indices.begin(), rand_indices.end(), g)

`cdFit` (`cd.cuh:164-166`) constructs `std::mt19937 g(rand())` and then
calls `initShuffle(ri, g)`, which RESEEDS `g` with the default
`random_state = 0`; the `rand()` seed is dead on arrival. So their
permutation sequence is a function of `std::mt19937(0)` and of
`std::shuffle`'s algorithm -- and THE STANDARD DOES NOT SPECIFY
`std::shuffle`'S ALGORITHM. libstdc++ and libc++ draw a different number
of variates per swap (libstdc++'s `__gen_two_uniform_ints` packs two draws
into one when the range allows), so the same seed gives a different
permutation on two conforming toolchains. A permutation that is not a
pure function of the seed cannot be gated bitwise and cannot be certified
across vendors.

THEREFORE `shuffle = true` (cuML Python `selection='random'`) IS REFUSED BY
NAME in both modes at `cd_fit`, and only `initShuffle`'s identity
permutation is here. DEVIATION 611 was reserved for an exact port
(`mt19937` is fully specified; libstdc++'s `std::shuffle` would have to be
transcribed and named as THE algorithm) and is NOT spent:
`solver/NOT_IMPLEMENTED.tsv` carries the entry.
"""


def init_shuffle(n_cols: Int) -> List[Int]:
    """`initShuffle`: `ri[i] = i`. The RNG reseed has no observable effect
    while `shuffle` is refused, and is therefore not mirrored."""
    var ri = List[Int]()
    ri.reserve(n_cols)
    for i in range(n_cols):
        ri.append(i)
    return ri^
