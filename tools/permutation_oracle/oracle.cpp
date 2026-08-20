// Oracle for gbdt/data/permutation.mojo.
//
// mersenne64.{h,cpp} are CatBoost's own files, byte for byte apart from the
// two util includes the shim replaces.  Everything below is transcribed from
// their headers at the cited lines, in C++, compiled by clang -- an
// independent implementation of the same algorithm, which is what makes it an
// oracle for the Mojo port rather than a second copy of it.
#include "mersenne64.h"
#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>

using NPrivate::TMersenne64;

// util/random/common_ops.h:48-60, GenUniform, with RandMax() = TResult(-1).
static ui64 GenUniform(ui64 max, TMersenne64& gen) {
    const ui64 randmax0 = (ui64)-1;
    const ui64 randmax = randmax0 - randmax0 % max;
    ui64 rand;
    while ((rand = gen.GenRand()) >= randmax) {
    }
    return rand % max;
}

// catboost/libs/helpers/cpu_random.h:17-33
struct TRandom {
    TMersenne64 Rng;
    explicit TRandom(ui64 seed) : Rng(seed) {}
    ui64 NextUniformL() { return Rng.GenRand(); }
    void Advance(ui32 n) { for (ui32 i = 0; i < n; ++i) NextUniformL(); }
    ui64 Uniform(ui64 size) { return GenUniform(size, Rng); }
};

// catboost/cuda/data/data_utils.h:21-47
static void Shuffle(ui64 seed, ui32 blockSize, ui32 sampleCount,
                    std::vector<ui32>* orderPtr) {
    TRandom rng(seed);
    rng.Advance(10);
    auto& order = *orderPtr;
    order.resize(sampleCount);
    std::iota(order.begin(), order.end(), 0u);

    if (blockSize == 1) {
        // util/random/shuffle.h:24-32
        const size_t sz = order.size();
        for (size_t i = 1; i < sz; ++i) {
            std::swap(order[i], order[rng.Uniform(i + 1)]);
        }
    } else {
        const ui32 blocksCount = (ui32)((order.size() + blockSize - 1) / blockSize);
        std::vector<ui32> blocks(blocksCount);
        std::iota(blocks.begin(), blocks.end(), 0u);
        for (size_t i = 1; i < blocks.size(); ++i) {
            std::swap(blocks[i], blocks[rng.Uniform(i + 1)]);
        }
        ui32 cursor = 0;
        for (ui32 i = 0; i < blocksCount; ++i) {
            const ui32 blockStart = blocks[i] * blockSize;
            const ui32 blockEnd = std::min<ui32>(blockStart + blockSize, (ui32)order.size());
            for (ui32 j = blockStart; j < blockEnd; ++j) {
                order[cursor++] = j;
            }
        }
    }
}

// catboost/cuda/data/permutation.h:93-95
static ui64 GetSeed(ui32 index, ui32 blockSize) {
    return 1664525ull * index + 1013904223ull + blockSize;
}

int main() {
    // 1. the raw MT19937-64 stream at three seeds, so a tempering or twist
    //    bug shows up before any permutation is built.
    ui64 seeds[3] = {5489ull, 19650218ull, GetSeed(3, 1)};
    printf("streams 3\n");
    for (int s = 0; s < 3; ++s) {
        TMersenne64 rng(seeds[s]);
        printf("stream %016llx 20\n", (unsigned long long)seeds[s]);
        for (int i = 0; i < 20; ++i) {
            // hex, because a ui64 above 2^63 does not survive a signed
            // decimal parse on the reading side
            printf("%016llx\n", (unsigned long long)rng.GenRand());
        }
    }

    // 1b. Uniform(t) itself, at a t where the REJECTION in GenUniform
    //     actually bites.  For t = 2^63 + 1 the acceptance bound is
    //     2^63 + 1, so about half of all draws are thrown away and a plain
    //     `GenRand() % t` diverges on the first rejected one.  At the sizes
    //     a permutation actually uses (a few thousand) rejection happens
    //     with probability ~ t / 2^64, so no fixture of a realistic size can
    //     tell the two apart -- which is why this row exists.
    struct TUniformCase { ui64 seed; ui64 size; int count; };
    TUniformCase uniformCases[] = {
        {GetSeed(3, 1), (1ull << 63) + 1ull, 24},
        {GetSeed(1, 1), 4096ull, 24},
        {GetSeed(2, 1), 3ull, 24},
    };
    const int nUniform = sizeof(uniformCases) / sizeof(uniformCases[0]);
    printf("uniforms %d\n", nUniform);
    for (int u = 0; u < nUniform; ++u) {
        TRandom rng(uniformCases[u].seed);
        rng.Advance(10);
        printf("uniform %016llx %016llx %d\n",
               (unsigned long long)uniformCases[u].seed,
               (unsigned long long)uniformCases[u].size,
               uniformCases[u].count);
        for (int i = 0; i < uniformCases[u].count; ++i) {
            printf("%016llx\n",
                   (unsigned long long)rng.Uniform(uniformCases[u].size));
        }
    }

    // 2. whole permutations, at the ids and sizes the CTR path uses.
    struct TCase { ui32 index; ui32 blockSize; ui32 n; };
    TCase cases[] = {
        {0, 1, 17}, {1, 1, 17}, {3, 1, 17},
        {1, 1, 1}, {1, 1, 2}, {1, 1, 313},
        {2, 1, 2003}, {3, 1, 2003},
        {1, 64, 2003}, {3, 64, 313},
    };
    const int nCases = sizeof(cases) / sizeof(cases[0]);
    printf("permutations %d\n", nCases);
    for (int c = 0; c < nCases; ++c) {
        std::vector<ui32> order;
        if (cases[c].index != 0) {
            Shuffle(GetSeed(cases[c].index, cases[c].blockSize),
                    cases[c].blockSize, cases[c].n, &order);
        } else {
            order.resize(cases[c].n);
            std::iota(order.begin(), order.end(), 0u);
        }
        printf("permutation %u %u %u\n", cases[c].index, cases[c].blockSize, cases[c].n);
        for (size_t i = 0; i < order.size(); ++i) {
            printf("%u\n", order[i]);
        }
    }
    return 0;
}
