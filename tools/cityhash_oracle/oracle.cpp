// Generates bench/cityhash_oracle.txt, read by `pixi run check-cityhash`.
//
// city.{h,cpp} and city_streaming.h in this directory are CatBoost's OWN
// files at 54a8143a (util/digest/), byte for byte apart from the util/
// includes that shim.h replaces. Their header states the trap outright:
// this is CityHash 1.0, whose results are DIFFERENT from mainline CityHash,
// so a port checked against any public CityHash vector would be checked
// against the wrong function.
//
// The two functions transcribed below are the whole call chain above it:
//   CalcCatFeatureHash  catboost/libs/cat_feature/cat_feature.cpp:6-8
//   CalcHash            catboost/libs/model/hash.h:11-14
// and the (ui64)(int) SIGN EXTENSION in the chain rows is
// catboost/libs/model/ctr_provider.h:107 (CalcHashes), where a ui32
// category hash >= 2^31 becomes 0xffffffff________ before entering
// CalcHash. A port that feeds the ui32 in unextended agrees on half of all
// hashes and silently disagrees on the other half.
#include "city.h"

#include <cstdio>
#include <string>
#include <vector>

static ui32 CalcCatFeatureHash(const char* s, size_t len) {
    return CityHash64(s, len) & 0xffffffff;
}

static ui64 CalcHash(ui64 a, ui64 b) {
    const ui64 MAGIC_MULT = 0x4906ba494954cb65ull;
    return MAGIC_MULT * (a + MAGIC_MULT * b);
}

int main() {
    std::vector<std::string> strings;

    // Realistic category spellings: integers-as-strings (amazon), words
    // with '-' and '?' (adult), the "nan" a missing value becomes, UTF-8,
    // an embedded NUL and a high byte (category strings are raw bytes).
    const char* fixed[] = {"", "a", "nan", "0", "1234", "39", "Private",
                           "United-States", "Never-married", "?",
                           "Some-college", ">50K"};
    for (const char* f : fixed)
        strings.push_back(std::string(f));
    strings.push_back(std::string("\x00\x01\xff", 3));
    strings.push_back(std::string("caf\xc3\xa9"));

    // One string per length that exercises a distinct branch of
    // CityHash64: 0 / 1..3 / 4..8 / 9..16 / 17..32 / 33..64, then >64 where
    // the 64-byte loop runs 1, 2, 4 and 15 times. Bytes come from a fixed
    // LCG; the check re-reads them from the hex dump, so both sides hash
    // identical inputs and the generator's quality is irrelevant.
    ui64 state = 0x123456789abcdef0ull;
    const size_t lens[] = {1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17,
                           24, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128,
                           129, 192, 255, 256, 300, 1000};
    for (size_t len : lens) {
        std::string s(len, '\0');
        for (size_t i = 0; i < len; ++i) {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            s[i] = (char)(unsigned char)(state >> 33);
        }
        strings.push_back(s);
    }

    printf("strings %zu\n", strings.size());
    for (const std::string& s : strings) {
        printf("str %zu ", s.size());
        if (s.empty())
            printf("-");
        for (unsigned char c : s)
            printf("%02x", c);
        printf(" %016llx %08x\n",
               (unsigned long long)CityHash64(s.data(), s.size()),
               CalcCatFeatureHash(s.data(), s.size()));
    }

    // The apply-time combination chains (ctr_provider.h:94-122): start at
    // 0, fold each element with CalcHash. Rows alternate the two element
    // kinds their CalcHashes feeds: a category hash sign-extended through
    // (ui64)(int), and a binary-split arm that is a bare 0 or 1.
    const size_t chain_lens[] = {1, 2, 3, 4, 8};
    printf("chains %zu\n", sizeof(chain_lens) / sizeof(chain_lens[0]));
    for (size_t k : chain_lens) {
        ui64 v = 0;
        printf("chain %zu ", k);
        for (size_t i = 0; i < k; ++i) {
            if (i % 3 == 2) {
                ui64 bit = i & 1;
                v = CalcHash(v, bit);
                printf("b%llu ", (unsigned long long)bit);
            } else {
                const std::string& s = strings[(7 * i + k) % strings.size()];
                ui32 h = CalcCatFeatureHash(s.data(), s.size());
                v = CalcHash(v, (ui64)(int)h);
                printf("h%08x ", h);
            }
        }
        printf("%016llx\n", (unsigned long long)v);
    }
    return 0;
}
