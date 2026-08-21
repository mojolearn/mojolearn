#pragma once
// Replaces the util/ includes of CatBoost's city.{h,cpp} at 54a8143a with
// the standard-library pieces they resolve to, so their files compile here
// unchanged. Everything below is a transcription of the util/ definition it
// stands in for, cited by file.
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <utility>

using ui8 = uint8_t;
using ui32 = uint32_t;
using ui64 = uint64_t;

// util/generic/strbuf.h: TStringBuf is a std::basic_string_view<char>
// descendant; city.h only calls .data() and .size().
using TStringBuf = std::string_view;

// util/generic/utility.h: DoSwap forwards to std::swap.
template <class T>
static inline void DoSwap(T& a, T& b) {
    std::swap(a, b);
}

// util/system/unaligned_mem.h:13-16: memcpy-based unaligned read,
// native (little-endian) byte order.
template <class T>
static inline T ReadUnaligned(const void* from) noexcept {
    T t;
    memcpy(&t, from, sizeof(T));
    return t;
}

#define Y_PURE_FUNCTION
#define Y_LIKELY(x) (x)
#define Y_UNLIKELY(x) (x)
#define Y_ASSERT(x) assert(x)
