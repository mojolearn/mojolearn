// Reference generator for extratrees/mojo_only/pcg_rng.mojo.
//
// This is RAFT's PCGenerator and cuML's fnv1a32, COPIED (not re-derived) with
// the CUDA decorations stripped, so that the numbers in pcg_reference.txt come
// from the upstream's own arithmetic rather than from our transcription of it.
// If our Mojo and this file agree, the Mojo is a faithful port; if they
// disagree, the Mojo is wrong.
//
// Pins:
//   RAFT 661a3b840c3300f95f053812a560c952c9d049a4
//     cpp/include/raft/random/detail/rng_device.cuh:546-683  (struct PCGenerator)
//     cpp/include/raft/random/detail/rng_device.cuh:173-183   (custom_next, UniformDistParams)
//     cpp/include/raft/random/detail/rng_device.cuh:185-206   (custom_next, UniformIntDistParams<_,uint32_t>)
//     cpp/include/raft/random/detail/rng_device.cuh:208-230   (custom_next, UniformIntDistParams<_,uint64_t>)
//     cpp/include/raft/util/integer_utils.hpp:207-237         (wmul_64bit)
//   cuML 00094f7e4e4b5da3a968d193a4da6085fa38f11b
//     cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels.cuh:100-113 (fnv1a32)
//     cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels.cuh:167-172 (the key chain + PCGenerator ctor)
//
// Build with tools/rng_oracle/build.sh. No CUDA toolkit is needed: every line
// below is plain C++ once HDI is defined away.

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>

#define HDI inline
#define DI inline

// ---------------------------------------------------------------------------
// raft/util/integer_utils.hpp:207-237, the non-__CUDA_ARCH__ branch verbatim.
// The __CUDA_ARCH__ branch is two PTX `mul.hi.u64` / `mul.lo.u64` instructions
// computing the same product.
// ---------------------------------------------------------------------------
HDI void wmul_64bit(uint64_t& res_hi, uint64_t& res_lo, uint64_t a, uint64_t b)
{
  uint32_t a_hi, a_lo, b_hi, b_lo;

  a_hi = uint32_t(a >> 32);
  a_lo = uint32_t(a & uint64_t(0x00000000FFFFFFFF));
  b_hi = uint32_t(b >> 32);
  b_lo = uint32_t(b & uint64_t(0x00000000FFFFFFFF));

  uint64_t t0 = uint64_t(a_lo) * uint64_t(b_lo);
  uint64_t t1 = uint64_t(a_hi) * uint64_t(b_lo);
  uint64_t t2 = uint64_t(a_lo) * uint64_t(b_hi);
  uint64_t t3 = uint64_t(a_hi) * uint64_t(b_hi);

  uint64_t carry = 0, trial = 0;

  res_lo = t0;
  trial  = res_lo + (t1 << 32);
  if (trial < res_lo) carry++;
  res_lo = trial;
  trial  = res_lo + (t2 << 32);
  if (trial < res_lo) carry++;
  res_lo = trial;

  // No need to worry about carry in this addition
  res_hi = (t1 >> 32) + (t2 >> 32) + t3 + carry;
}

// ---------------------------------------------------------------------------
// raft/random/detail/rng_device.cuh:546-683, verbatim minus the `half` members
// and the DeviceState-taking constructor (neither has a Mojo counterpart).
// ---------------------------------------------------------------------------
struct PCGenerator {
  HDI PCGenerator(uint64_t seed, uint64_t subsequence, uint64_t offset)
  {
    _init_pcg(seed, subsequence, offset);
  }

  // Based on "Random Number Generation with Arbitrary Strides" F. B. Brown
  HDI void skipahead(uint64_t offset)
  {
    uint64_t G = 1;
    uint64_t h = 6364136223846793005ULL;
    uint64_t C = 0;
    uint64_t f = inc;
    while (offset) {
      if (offset & 1) {
        G = G * h;
        C = C * h + f;
      }
      f = f * (h + 1);
      h = h * h;
      offset >>= 1;
    }
    pcg_state = pcg_state * G + C;
  }

  HDI uint32_t next_u32()
  {
    uint32_t ret;
    uint64_t oldstate   = pcg_state;
    pcg_state           = oldstate * 6364136223846793005ULL + inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot        = oldstate >> 59u;
    ret                 = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    return ret;
  }
  HDI uint64_t next_u64()
  {
    uint64_t ret;
    uint32_t a, b;
    a   = next_u32();
    b   = next_u32();
    ret = uint64_t(a) | (uint64_t(b) << 32);
    return ret;
  }

  HDI int32_t next_i32()
  {
    int32_t ret;
    uint32_t val;
    val = next_u32();
    ret = int32_t(val & 0x7fffffff);
    return ret;
  }

  HDI int64_t next_i64()
  {
    int64_t ret;
    uint64_t val;
    val = next_u64();
    ret = int64_t(val & 0x7fffffffffffffff);
    return ret;
  }

  HDI float next_float()
  {
    float ret;
    uint32_t val = next_u32() >> 8;
    ret          = static_cast<float>(val) / (1U << 24);
    return ret;
  }

  HDI double next_double()
  {
    double ret;
    uint64_t val = next_u64() >> 11;
    ret          = static_cast<double>(val) / (1LU << 53);
    return ret;
  }

  HDI void next(uint32_t& ret) { ret = next_u32(); }
  HDI void next(uint64_t& ret) { ret = next_u64(); }
  HDI void next(int32_t& ret) { ret = next_i32(); }
  HDI void next(int64_t& ret) { ret = next_i64(); }

  HDI void next(float& ret) { ret = next_float(); }
  HDI void next(double& ret) { ret = next_double(); }

 private:
  HDI void _init_pcg(uint64_t seed, uint64_t subsequence, uint64_t offset)
  {
    pcg_state = uint64_t(0);
    inc       = (subsequence << 1u) | 1u;
    uint32_t discard;
    next(discard);
    pcg_state += seed;
    next(discard);
    skipahead(offset);
  }
  uint64_t pcg_state;
  uint64_t inc;
};

// raft/random/detail/rng_device.cuh:58-69
template <typename OutType>
struct UniformDistParams {
  OutType start;
  OutType end;
};

template <typename OutType, typename DiffType>
struct UniformIntDistParams {
  OutType start;
  OutType end;
  DiffType diff;
};

// raft/random/detail/rng_device.cuh:173-183
template <typename GenType, typename OutType, typename LenType>
HDI void custom_next(GenType& gen,
                     OutType* val,
                     UniformDistParams<OutType> params,
                     LenType idx    = 0,
                     LenType stride = 0)
{
  OutType res;
  gen.next(res);
  // DEVIATION 142, amended: an explicit fma, matching the Mojo side. RAFT
  // writes `(res * (end - start)) + start`, which nvcc contracts into exactly
  // this under its default --fmad=true, so the fused form is what the upstream
  // computes on the hardware they ship for -- and it is the only form a GPU
  // backend can be made to hold (six source-level barriers were measured and
  // every one fused anyway).
  *val = std::fma(res, params.end - params.start, params.start);
}

// raft/random/detail/rng_device.cuh:185-206
template <typename GenType, typename OutType, typename LenType>
HDI void custom_next(GenType& gen,
                     OutType* val,
                     UniformIntDistParams<OutType, uint32_t> params,
                     LenType idx    = 0,
                     LenType stride = 0)
{
  uint32_t x = 0;
  uint32_t s = params.diff;
  gen.next(x);
  uint64_t m = uint64_t(x) * s;
  uint32_t l = uint32_t(m);
  if (l < s) {
    uint32_t t = (-s) % s;  // (2^32 - s) mod s
    while (l < t) {
      gen.next(x);
      m = uint64_t(x) * s;
      l = uint32_t(m);
    }
  }
  *val = OutType(m >> 32) + params.start;
}

// raft/random/detail/rng_device.cuh:208-230
template <typename GenType, typename OutType, typename LenType>
HDI void custom_next(GenType& gen,
                     OutType* val,
                     UniformIntDistParams<OutType, uint64_t> params,
                     LenType idx    = 0,
                     LenType stride = 0)
{
  uint64_t x = 0;
  gen.next(x);
  uint64_t s = params.diff;
  uint64_t m_lo, m_hi;
  // m = x * s;
  wmul_64bit(m_hi, m_lo, x, s);
  if (m_lo < s) {
    uint64_t t = (-s) % s;  // (2^64 - s) mod s
    while (m_lo < t) {
      gen.next(x);
      wmul_64bit(m_hi, m_lo, x, s);
    }
  }
  *val = OutType(m_hi) + params.start;
}

// ---------------------------------------------------------------------------
// cuML kernels/builder_kernels.cuh:100-113
// ---------------------------------------------------------------------------
const uint32_t fnv1a32_prime = uint32_t(16777619);
const uint32_t fnv1a32_basis = uint32_t(2166136261);
HDI uint32_t fnv1a32(uint32_t hash, uint32_t txt)
{
  hash ^= (txt >> 0) & 0xFF;
  hash *= fnv1a32_prime;
  hash ^= (txt >> 8) & 0xFF;
  hash *= fnv1a32_prime;
  hash ^= (txt >> 16) & 0xFF;
  hash *= fnv1a32_prime;
  hash ^= (txt >> 24) & 0xFF;
  hash *= fnv1a32_prime;
  return hash;
}

// cuML kernels/builder_kernels.cuh:167-170, with the extra `feature` component
// that is OUR deviation 130. Order mirrors theirs: the per-candidate component
// first (they use threadIdx.x), then treeid, then nodeid.
static uint64_t key_chain(uint32_t feature, uint32_t tree, uint32_t node)
{
  uint64_t subsequence(fnv1a32_basis);
  subsequence = fnv1a32(uint32_t(subsequence), feature);
  subsequence = fnv1a32(uint32_t(subsequence), tree);
  subsequence = fnv1a32(uint32_t(subsequence), node);
  return subsequence;
}

// ---------------------------------------------------------------------------
// dump
// ---------------------------------------------------------------------------
static uint32_t f32bits(float f)
{
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

static void print_f32(FILE* out, const char* tag, float f)
{
  // A float persisted as decimal alone is not evidence (Mojo's String(Float32)
  // does not round-trip). Decimal is for humans; the hex is what is compared.
  fprintf(out, "%s %.9g/%08x\n", tag, double(f), f32bits(f));
}

struct Triple {
  uint64_t seed, subsequence, offset;
  const char* note;
};

int main()
{
  FILE* out = fopen("pcg_reference.txt", "w");
  if (!out) {
    fprintf(stderr, "cannot open pcg_reference.txt for writing\n");
    return 2;
  }

  fprintf(out, "# generated by extratrees/tools/rng_oracle/main.cpp -- do not hand-edit\n");
  fprintf(out, "raft_pin 661a3b840c3300f95f053812a560c952c9d049a4\n");
  fprintf(out, "cuml_pin 00094f7e4e4b5da3a968d193a4da6085fa38f11b\n");

  // --- fnv1a32, one step at a time -----------------------------------------
  {
    const uint32_t hashes[] = {fnv1a32_basis, 0u, 1u, 0xFFFFFFFFu, 0x811C9DC5u, 0xDEADBEEFu};
    const uint32_t txts[]   = {0u, 1u, 255u, 256u, 0x0000FF00u, 0x12345678u, 0xFFFFFFFFu, 4242u};
    int n = int(sizeof(hashes) / sizeof(hashes[0])) * int(sizeof(txts) / sizeof(txts[0]));
    fprintf(out, "fnv_cases %d\n", n);
    for (uint32_t h : hashes)
      for (uint32_t t : txts)
        fprintf(out, "fnv %08x %08x %08x\n", h, t, fnv1a32(h, t));
  }

  // --- the (feature, tree, node) key chain ---------------------------------
  struct Key {
    uint32_t feature, tree, node;
  };
  const Key keys[] = {
    {0, 0, 0},
    {7, 3, 11},
    {19, 257, 65535},
    {0xFFFFFFFFu, 12345, 6},
    {2, 2, 2},
    {1023, 0, 4095},
    {5, 900001, 33},
  };
  {
    int n = int(sizeof(keys) / sizeof(keys[0]));
    fprintf(out, "chain_cases %d\n", n);
    for (const Key& k : keys)
      fprintf(out,
              "chain %08x %08x %08x %016llx\n",
              k.feature,
              k.tree,
              k.node,
              (unsigned long long)key_chain(k.feature, k.tree, k.node));
  }

  // --- streams --------------------------------------------------------------
  const Triple triples[] = {
    {0ull, 0ull, 0ull, "all-zero"},
    {0xDEADBEEFCAFEBABEull, 1ull, 0ull, "round subsequence"},
    {42ull, key_chain(7, 3, 11), 0ull, "hashed subsequence"},
    {42ull, key_chain(0, 0, 0), 5ull, "hashed subsequence, small offset"},
    {0x1234567890ABCDEFull, key_chain(19, 257, 65535), 1ull, "hashed, offset 1"},
    {1ull, key_chain(0xFFFFFFFFu, 12345, 6), 1023ull, "hashed, offset 1023"},
    {0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFull, "extremes"},
    {7ull, key_chain(2, 2, 2), 64ull, "hashed, power-of-two offset"},
    {0x9E3779B97F4A7C15ull, key_chain(5, 900001, 33), 1000000ull, "hashed, big offset"},
  };
  const int n_triples = int(sizeof(triples) / sizeof(triples[0]));

  {
    fprintf(out, "streams %d\n", n_triples);
    for (const Triple& tr : triples) {
      const int n_draws = 16;
      fprintf(out,
              "stream %016llx %016llx %016llx %d\n",
              (unsigned long long)tr.seed,
              (unsigned long long)tr.subsequence,
              (unsigned long long)tr.offset,
              n_draws);
      PCGenerator gen(tr.seed, tr.subsequence, tr.offset);
      for (int i = 0; i < n_draws; ++i)
        fprintf(out, "u32 %08x\n", gen.next_u32());
    }
  }

  // --- next_u64 -------------------------------------------------------------
  {
    fprintf(out, "u64_streams %d\n", n_triples);
    for (const Triple& tr : triples) {
      const int n_draws = 8;
      fprintf(out,
              "u64stream %016llx %016llx %016llx %d\n",
              (unsigned long long)tr.seed,
              (unsigned long long)tr.subsequence,
              (unsigned long long)tr.offset,
              n_draws);
      PCGenerator gen(tr.seed, tr.subsequence, tr.offset);
      for (int i = 0; i < n_draws; ++i)
        fprintf(out, "u64 %016llx\n", (unsigned long long)gen.next_u64());
    }
  }

  // --- uniform ints, 32-bit diff -------------------------------------------
  {
    struct R32 {
      uint32_t start, diff;
    };
    // 1 and 2 are the degenerate ends; 1000003 and 97 are prime; 4096 is a
    // power of two; 0xFFFFFFFF is the whole range minus one.
    const R32 ranges[] = {{0u, 1u},
                          {0u, 2u},
                          {0u, 10u},
                          {0u, 97u},
                          {5u, 4096u},
                          {0u, 1000003u},
                          {1000u, 3u},
                          {0u, 0xFFFFFFFFu}};
    int n = n_triples * int(sizeof(ranges) / sizeof(ranges[0]));
    fprintf(out, "uint32_cases %d\n", n);
    for (const Triple& tr : triples) {
      for (const R32& r : ranges) {
        const int n_draws = 12;
        fprintf(out,
                "uint32 %016llx %016llx %016llx %u %u %d\n",
                (unsigned long long)tr.seed,
                (unsigned long long)tr.subsequence,
                (unsigned long long)tr.offset,
                r.start,
                r.diff,
                n_draws);
        PCGenerator gen(tr.seed, tr.subsequence, tr.offset);
        UniformIntDistParams<uint32_t, uint32_t> p;
        p.start = r.start;
        p.end   = r.start + r.diff;
        p.diff  = r.diff;
        for (int i = 0; i < n_draws; ++i) {
          uint32_t v = 0;
          custom_next(gen, &v, p, 0, 0);
          fprintf(out, "i32 %u\n", v);
        }
      }
    }
  }

  // --- uniform ints, 64-bit diff (this is the overload cuML's sampler uses) --
  {
    struct R64 {
      uint64_t start, diff;
    };
    const R64 ranges[] = {{0ull, 1ull},
                          {0ull, 3ull},
                          {0ull, 100ull},
                          {0ull, 1000003ull},
                          {17ull, 65536ull},
                          {0ull, 12345678901ull},
                          {0ull, 0x8000000000000001ull}};
    int n = n_triples * int(sizeof(ranges) / sizeof(ranges[0]));
    fprintf(out, "uint64_cases %d\n", n);
    for (const Triple& tr : triples) {
      for (const R64& r : ranges) {
        const int n_draws = 12;
        fprintf(out,
                "uint64 %016llx %016llx %016llx %llu %llu %d\n",
                (unsigned long long)tr.seed,
                (unsigned long long)tr.subsequence,
                (unsigned long long)tr.offset,
                (unsigned long long)r.start,
                (unsigned long long)r.diff,
                n_draws);
        PCGenerator gen(tr.seed, tr.subsequence, tr.offset);
        UniformIntDistParams<uint64_t, uint64_t> p;
        p.start = r.start;
        p.end   = r.start + r.diff;
        p.diff  = r.diff;
        for (int i = 0; i < n_draws; ++i) {
          uint64_t v = 0;
          custom_next(gen, &v, p, 0, 0);
          fprintf(out, "i64 %llu\n", (unsigned long long)v);
        }
      }
    }
  }

  // --- raw next_float -------------------------------------------------------
  {
    fprintf(out, "float_streams %d\n", n_triples);
    for (const Triple& tr : triples) {
      const int n_draws = 12;
      fprintf(out,
              "floatstream %016llx %016llx %016llx %d\n",
              (unsigned long long)tr.seed,
              (unsigned long long)tr.subsequence,
              (unsigned long long)tr.offset,
              n_draws);
      PCGenerator gen(tr.seed, tr.subsequence, tr.offset);
      for (int i = 0; i < n_draws; ++i)
        print_f32(out, "f", gen.next_float());
    }
  }

  // --- uniform floats over [start, end) -------------------------------------
  {
    struct RF {
      float start, end;
    };
    const RF ranges[] = {{0.0f, 1.0f},
                         {-3.5f, 2.25f},
                         {-1.0f, -1.0f},
                         {1.0e-8f, 1.0000001e-8f},
                         {-1.0e30f, 1.0e30f},
                         {0.1f, 0.30000001192092896f},
                         {12345.678f, 12345.679f}};
    int n = n_triples * int(sizeof(ranges) / sizeof(ranges[0]));
    fprintf(out, "ufloat_cases %d\n", n);
    for (const Triple& tr : triples) {
      for (const RF& r : ranges) {
        const int n_draws = 10;
        fprintf(out,
                "ufloat %016llx %016llx %016llx %.9g/%08x %.9g/%08x %d\n",
                (unsigned long long)tr.seed,
                (unsigned long long)tr.subsequence,
                (unsigned long long)tr.offset,
                double(r.start),
                f32bits(r.start),
                double(r.end),
                f32bits(r.end),
                n_draws);
        PCGenerator gen(tr.seed, tr.subsequence, tr.offset);
        UniformDistParams<float> p;
        p.start = r.start;
        p.end   = r.end;
        for (int i = 0; i < n_draws; ++i) {
          float v = 0.0f;
          custom_next(gen, &v, p, 0, 0);
          print_f32(out, "uf", v);
        }
      }
    }
  }

  fprintf(out, "end\n");
  fclose(out);
  fprintf(stderr, "wrote pcg_reference.txt\n");
  return 0;
}
