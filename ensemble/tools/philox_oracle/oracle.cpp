// Dumps what RAFT's own `uniformInt` under `GenPhilox` produces, so the Mojo
// port can be diffed against it VALUE FOR VALUE rather than against our
// reading of it.
//
// WHY THIS FILE EXISTS. cuML's GPU Random Forest draws every tree's bootstrap
// row sample with exactly one call (`cpp/src/randomforest/randomforest.cuh:
// 140-142`):
//
//     raft::random::uniformInt<int>(
//         stream_resources, rng_state, selected_rows.data(),
//         selected_rows.size(), 0, n_rows_);
//
// with `rng_state = RngState(rs, GenPhilox)` and
// `rs = fnv1a32(fnv1a32(fnv1a32_basis, seed), tree_id)` (`:119-123`).
// Those row ids ARE which rows every tree sees. One wrong value and every
// tree in the forest differs, and no downstream check can attribute the
// difference, because a wrong bootstrap sample is still a perfectly plausible
// bootstrap sample. So the reference values come from THEIR generator,
// compiled and run. Same discipline and the same shape as
// `ensemble/tools/shuffle_oracle/`, which compiles CCCL's shuffle_iterator for
// the same reason, and as `tools/permutation_oracle/`, which compiles
// CatBoost's Mersenne twister.
//
// WHAT IS COMPILED AND WHAT IS TRANSCRIBED. `build.sh` explains the split and
// why; in one line: the Philox rounds and the cuRAND state machine below are
// NVIDIA's own bytes, fetched at build time and never committed (proprietary
// licence); RAFT's thin layer on top is Apache-2.0 and is transcribed here,
// cited line by line, because RAFT's RNG entry points are `DI`
// (__device__ __forceinline__) and pull in cuda_runtime.
//
// No CUDA toolkit and no GPU are required. Output goes to
// `ensemble/bench/philox_oracle.txt`, which IS COMMITTED -- regenerate it when
// the pin moves and read the diff, because that diff is the behaviour change.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

// --- shims: the three CUDA vocabulary types cuRAND's Philox header needs ---
// `curand_philox4x32_x.h` is the D. E. Shaw Research Random123 construction
// under a BSD-3 notice inside an NVIDIA-proprietary file. It is
// host-and-device (`QUALIFIERS` defaults to `static __forceinline__
// __device__`, and we redefine it), so all it wants from CUDA is uint2/uint4.
struct uint2 {
  unsigned int x, y;
};
struct uint4 {
  unsigned int x, y, z, w;
};
static inline uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
  uint4 r;
  r.x = x;
  r.y = y;
  r.z = z;
  r.w = w;
  return r;
}
#define QUALIFIERS static inline

#include <curand_philox4x32_x.h>  // NVIDIA's, unmodified, fetched by build.sh
#include "curand_philox_state.inc"  // sliced out of NVIDIA's curand_kernel.h

#ifndef ORACLE_CURAND_VERSION
#define ORACLE_CURAND_VERSION "unknown"
#endif

// ===========================================================================
// RAFT, transcribed (Apache-2.0). Pin: rapidsai/raft v26.08.00,
// ebf92684b8a15addcddb43f442a382d528e8bd77.
// ===========================================================================

// `raft/random/detail/rng_device.cuh:426-533` -- PhiloxGenerator. The only
// members that matter here are the DeviceState constructor (`:440-443`) and
// `next_u32` (`:449-453`); `next(uint32_t&)` at `:523` is what `custom_next`'s
// `gen.next(x)` resolves to, and it is `next_u32` verbatim.
//
//     DI PhiloxGenerator(const DeviceState<PhiloxGenerator>& rng_state,
//                        const uint64_t subsequence)
//     { curand_init(rng_state.seed, rng_state.base_subsequence + subsequence,
//                   0, &philox_state); }
//     DI uint32_t next_u32() { return curand(&(this->philox_state)); }
//
// Note the OFFSET IS ALWAYS 0 on this path, and the subsequence is
// `base_subsequence + subsequence` -- a 64-bit add that is allowed to wrap.
struct PhiloxGenerator {
  curandStatePhilox4_32_10_t philox_state;

  PhiloxGenerator(uint64_t seed, uint64_t base_subsequence, uint64_t subsequence)
  {
    curand_init(seed, base_subsequence + subsequence, 0, &philox_state);
  }

  uint32_t next_u32() { return curand(&philox_state); }
};

// A scripted stand-in for the generator, so the RANGE REDUCTION can be
// compared on its own. RAFT's `custom_next` is a template over `GenType`
// (`rng_device.cuh:175`), so substituting a source here is their own
// parameterisation, not a rewrite of the function under test.
struct ScriptedGen {
  const uint32_t* xs;
  int n;
  int i;
  uint32_t next_u32()
  {
    uint32_t v = xs[i < n ? i : n - 1];
    i++;
    return v;
  }
};

// `rng_device.cuh:53-58`
//   template <typename OutType, typename DiffType>
//   struct UniformIntDistParams { OutType start; OutType end; DiffType diff; };
//
// `rng_impl.cuh:88-107` fills it. For `sizeof(OutType) == 4` -- which is
// cuML's `int` -- DiffType is uint32_t and `diff = uint32_t(end - start)`.
// `end` is stored and NEVER READ by the reduction.

// `rng_device.cuh:175-196` -- custom_next, the uint32 arm, transcribed
// statement for statement. This is Lemire's nearly-divisionless bounded draw.
// `draws` is ours: it counts generator calls so the rejection loop can be
// held to a count and not only to a value.
template <typename Gen>
static int custom_next_uniform_int_u32(Gen& gen, int start, uint32_t diff, int* draws)
{
  uint32_t x = 0;
  uint32_t s = diff;
  x = gen.next_u32();
  (*draws)++;
  uint64_t m = uint64_t(x) * s;
  uint32_t l = uint32_t(m);
  if (l < s) {
    uint32_t t = (-s) % s;  // (2^32 - s) mod s
    while (l < t) {
      x = gen.next_u32();
      (*draws)++;
      m = uint64_t(x) * s;
      l = uint32_t(m);
    }
  }
  return int(m >> 32) + start;
}

// `rng_device.cuh:675-694` (rngKernel<1>) plus `rng_impl.cuh:64-74`
// (call_rng_kernel), on the host.
//
//     LenType tid = threadIdx.x + LenType(blockIdx.x) * blockDim.x;
//     GenType gen(rng_state, (uint64_t)tid);
//     const LenType stride = gridDim.x * blockDim.x;
//     for (LenType idx = tid; idx < len; idx += stride * ITEMS_PER_CALL) {
//       OutType val[ITEMS_PER_CALL];
//       custom_next(gen, val, params, idx, stride);
//       for (int i = 0; i < ITEMS_PER_CALL; i++)
//         if ((idx + i * stride) < len) ptr[idx + i * stride] = val[i];
//     }
//
// ITEMS_PER_CALL is 1 for uniformInt (`rng_impl.cuh:99`, `:105`), so it is a
// plain grid-stride loop: thread `tid` writes `tid, tid+stride, tid+2*stride,
// ...` from ITS OWN generator, whose subsequence is `base_subsequence + tid`.
//
// STRIDE IS `gridDim.x * blockDim.x`, AND THEIR LAUNCH SIZES IT FROM THE
// HARDWARE: `n_threads = 256`, `n_blocks = 4 * getMultiProcessorCount()`
// (`rng_impl.cuh:70-71`; `getMultiProcessorCount` reads
// cudaDevAttrMultiProcessorCount, `util/cudart_utils.hpp:301-308`). So RAFT's
// output for a given seed IS A FUNCTION OF THE GPU MODEL. That is not a
// caveat about this oracle -- it is a property of theirs, and the table takes
// `stride` as an explicit parameter because of it.
static void fill_uniform_int(uint64_t seed,
                             uint64_t base_sub,
                             uint64_t stride,
                             int* out,
                             size_t len,
                             int start,
                             uint32_t diff)
{
  for (uint64_t tid = 0; tid < stride; tid++) {
    PhiloxGenerator gen(seed, base_sub, tid);
    int draws = 0;
    for (size_t idx = (size_t)tid; idx < len; idx += (size_t)stride) {
      out[idx] = custom_next_uniform_int_u32(gen, start, diff, &draws);
    }
  }
}

// cuML `cpp/src/decisiontree/batched-levelalgo/random_utils.cuh:17-31`
static const uint32_t kFnvPrime = 16777619u;
static const uint32_t kFnvBasis = 2166136261u;

static uint32_t fnv1a32(uint32_t hash, uint32_t txt)
{
  hash ^= (txt >> 0) & 0xFF;
  hash *= kFnvPrime;
  hash ^= (txt >> 8) & 0xFF;
  hash *= kFnvPrime;
  hash ^= (txt >> 16) & 0xFF;
  hash *= kFnvPrime;
  hash ^= (txt >> 24) & 0xFF;
  hash *= kFnvPrime;
  return hash;
}

// `randomforest.cuh:120-122`. NOTE THE TRUNCATION: `seed_` is `uint64_t`
// (`:217`) and `fnv1a32` takes `uint32_t`, and this call site uses `fnv1a32`
// DIRECTLY rather than `fnv1a32_combine`, so the high 32 bits of the user's
// seed are silently discarded. Theirs; transcribed, not corrected.
static uint32_t rng_seed_for_tree(uint64_t seed, int tree_id)
{
  uint32_t rs = kFnvBasis;
  rs          = fnv1a32(rs, (uint32_t)seed);
  rs          = fnv1a32(rs, (uint32_t)tree_id);
  return rs;
}

// ===========================================================================

struct SeedSub {
  uint64_t seed;
  uint64_t sub;
};

// Subsequences chosen to walk the carry in `Philox_State_Incr_hi`, which is
// where curand_init puts the subsequence: ctr.z += lo32(n), ctr.w += hi32(n).
static const SeedSub kSeedSubs[] = {
  {0ull, 0ull},
  {0ull, 1ull},
  {1ull, 0ull},
  {12345ull, 0ull},
  {12345ull, 1ull},
  {12345ull, 2ull},
  {12345ull, 3ull},
  {12345ull, 255ull},
  {12345ull, 4294967295ull},   // ctr.z = 0xFFFFFFFF, ctr.w = 0
  {12345ull, 4294967296ull},   // ctr.z = 0,          ctr.w = 1
  {12345ull, 4294967297ull},
  {12345ull, 18446744073709551615ull},
  {3735928559ull, 7ull},
  {16045690984503098046ull, 0ull},  // 0xDEADBEEFCAFEBABE: key.y is nonzero
  {16045690984503098046ull, 110591ull},
};

int main()
{
  printf("# RAFT uniformInt<int> under GenPhilox -- reference values\n");
  printf("# RAFT v26.08.00 (ebf92684b8a15addcddb43f442a382d528e8bd77), the RAFT\n");
  printf("# cuML v26.08.00 (265b9da6a0e75dbef071a3168398b993a5ff6f0e) resolves.\n");
  printf("# Philox core + cuRAND state machine compiled from cuRAND %s.\n",
         ORACLE_CURAND_VERSION);
  printf("# generated by ensemble/tools/philox_oracle/oracle.cpp -- do not hand-edit\n");

  // --- layer 1: the raw counter stream ----------------------------------
  // `ctr <seed> <sub> ctr.x ctr.y ctr.z ctr.w key.x key.y out.x out.y out.z out.w STATE`
  // The state immediately after `curand_init(seed, sub, 0, &s)`, i.e. exactly
  // what RAFT's DeviceState constructor (`rng_device.cuh:440-443`) leaves
  // behind. Separated out because it is the single place a port can put the
  // subsequence in the wrong counter word and still produce a uniform stream.
  printf("section ctr\n");
  for (const SeedSub& c : kSeedSubs) {
    curandStatePhilox4_32_10_t s;
    curand_init(c.seed, c.sub, 0, &s);
    printf("ctr %llu %llu %u %u %u %u %u %u %u %u %u %u %u\n",
           (unsigned long long)c.seed,
           (unsigned long long)c.sub,
           s.ctr.x, s.ctr.y, s.ctr.z, s.ctr.w,
           s.key.x, s.key.y,
           s.output.x, s.output.y, s.output.z, s.output.w,
           s.STATE);
  }

  // --- layer 2: the raw 32-bit draw stream ------------------------------
  // `raw <seed> <sub> <count> v0 .. v(count-1)`
  // 13 draws crosses three 4-wide Philox blocks, so it exercises the
  // `STATE == 4 -> increment the counter and re-run the 10 rounds` edge
  // twice. A port that emits the four words of a block in the wrong order,
  // or that forgets to bump the counter, fails here and nowhere earlier.
  printf("section raw\n");
  for (const SeedSub& c : kSeedSubs) {
    curandStatePhilox4_32_10_t s;
    curand_init(c.seed, c.sub, 0, &s);
    printf("raw %llu %llu 13", (unsigned long long)c.seed, (unsigned long long)c.sub);
    for (int i = 0; i < 13; i++) printf(" %u", curand(&s));
    printf("\n");
  }

  // --- layer 3: the range reduction, on its own -------------------------
  // `lemire <s> <start> <nx> x0..x(nx-1) <val> <draws>`
  // Driven by a scripted draw stream so the reduction is isolated from the
  // generator. x = 0 is the universal adversary: it makes `l == 0`, which is
  // below `s` for every legal `s` and below `t` whenever `t > 0`, so a
  // leading run of zeros forces the rejection loop to spin a known number of
  // times. THE REJECTION LOOP IS NOT REACHABLE AT CUML'S CALL SITE (start=0,
  // end=n_rows, so t = 2^32 mod n_rows and the entry probability is
  // t/2^32 < 2.4e-10 for any n_rows they can hold in memory) -- it is
  // transcribed anyway, and reached deliberately here, because an unreached
  // branch is an unchecked branch even when the reason it is unreached is
  // arithmetic.
  //
  // The `s` values are adversarial on purpose: a power of two (t == 0, so
  // the `if` is entered and the `while` still never runs), one either side
  // of a power of two, a range of 1, INT_MAX, a diff that only a negative
  // `start` can produce and whose rejection rate is ~1/2, and the full
  // 2^32-1 span.
  printf("section lemire\n");
  {
    struct Case {
      uint32_t s;
      int start;
      int nx;
      uint32_t xs[6];
    };
    const Case cases[] = {
      // range of 1: every draw must return `start`
      {1u, 0, 3, {0u, 0u, 4294967295u}},
      {1u, -7, 3, {123456789u, 0u, 1u}},
      // exact power of two: t == 0, the while loop cannot run
      {1024u, 0, 3, {0u, 4194304u, 3735928559u}},
      {1024u, 0, 2, {4194304u, 12345u}},
      // one below a power of two: t == 4
      {1023u, 0, 5, {0u, 0u, 0u, 123456789u, 1u}},
      // one above a power of two
      {1025u, 0, 4, {0u, 0u, 2863311530u, 5u}},
      // small odd ranges, heavy rejection pressure relative to their size
      {3u, 0, 4, {0u, 0u, 1u, 2u}},
      {7u, 0, 4, {0u, 3735928559u, 1u, 2u}},
      // a realistic row count
      {1000000u, 0, 3, {0u, 3141592653u, 2718281828u}},
      // INT_MAX: the largest diff cuML's own `uniformInt<int>` can reach
      // with start = 0
      {2147483647u, 0, 4, {0u, 0u, 1u, 2863311530u}},
      // 2^31: a power of two again, but at the top of the range
      {2147483648u, 0, 3, {0u, 1u, 4294967295u}},
      // 2^31 + 1: t == 2^31 - 1, so ~half of all draws are rejected. Only a
      // NEGATIVE start can produce this diff from an int pair
      // (start = -2, end = INT_MAX). Note the return value overflows int and
      // wraps; see -fwrapv in build.sh.
      {2147483649u, -2, 5, {0u, 0u, 0u, 1u, 2u}},
      {2147483649u, -2, 3, {2147483648u, 0u, 0u}},
      // the full 32-bit span, which their `uint32_t(end - start)` produces
      // from (INT_MIN, INT_MAX) by signed overflow
      {4294967295u, -2147483648, 4, {0u, 0u, 1u, 2863311530u}},
    };
    for (const Case& c : cases) {
      ScriptedGen g{c.xs, c.nx, 0};
      int draws = 0;
      int val   = custom_next_uniform_int_u32(g, c.start, c.s, &draws);
      printf("lemire %u %d %d", c.s, c.start, c.nx);
      for (int i = 0; i < c.nx; i++) printf(" %u", c.xs[i]);
      printf(" %d %d\n", val, draws);
    }
  }

  // --- layer 4: uniformInt from a real generator, one thread's worth -----
  // `uint <seed> <sub> <start> <end> <count> v0 .. v(count-1)`
  // Ten CONSECUTIVE reduced draws off one PhiloxGenerator, which is what a
  // single thread produces across loop iterations when `len > stride`. This
  // is the layer that ties layers 2 and 3 together without the index mapping
  // in the way.
  printf("section uint\n");
  {
    struct Case {
      uint64_t seed;
      uint64_t sub;
      int start;
      int end;
    };
    const Case cases[] = {
      {12345ull, 0ull, 0, 2},
      {12345ull, 0ull, 0, 3},
      {12345ull, 0ull, 0, 1000},
      {12345ull, 7ull, 0, 1000000},
      {12345ull, 0ull, 0, 2147483647},
      {12345ull, 0ull, -2, 2147483647},
      {16045690984503098046ull, 3ull, -50, 50},
      {0ull, 0ull, 0, 1},
      {2166136261ull, 110591ull, 0, 464809},
    };
    for (const Case& c : cases) {
      PhiloxGenerator gen(c.seed, 0ull, c.sub);
      uint32_t diff = (uint32_t)(c.end - c.start);
      int draws     = 0;
      printf("uint %llu %llu %d %d 10",
             (unsigned long long)c.seed, (unsigned long long)c.sub, c.start, c.end);
      for (int i = 0; i < 10; i++)
        printf(" %d", custom_next_uniform_int_u32(gen, c.start, diff, &draws));
      printf("\n");
    }
  }

  // --- layer 5: the whole array, with the index mapping ------------------
  // `fill <seed> <base_sub> <stride> <len> <start> <end> v0 .. v(len-1)`
  // THIS IS THE LAYER THAT MATTERS, and the one a distributional test cannot
  // see: which thread writes which index, and which draw off which
  // subsequence lands there.
  //
  // `len` values are deliberately NOT multiples of 4 or of 256. There is no
  // packing to get wrong -- Philox emits four 32-bit words per counter block
  // and `curand()` hands them out one at a time, so a `len` that is not a
  // multiple of 4 simply leaves the tail of the last block unconsumed -- but
  // a port that tried to vectorise the block would break exactly here.
  //
  // Small strides are the point of the last four rows: with stride >= len
  // every thread does one draw and the loop never runs, which would leave
  // the grid-stride mapping completely unchecked.
  printf("section fill\n");
  {
    struct Case {
      uint64_t seed;
      uint64_t base_sub;
      uint64_t stride;
      size_t len;
      int start;
      int end;
    };
    const Case cases[] = {
      {12345ull, 0ull, 110592ull, 17, 0, 100},
      {12345ull, 0ull, 110592ull, 251, 0, 1000},
      {2166136261ull, 0ull, 110592ull, 1000, 0, 1000},
      {12345ull, 0ull, 110592ull, 63, 0, 1},
      // stride < len: the grid-stride loop actually runs
      {12345ull, 0ull, 512ull, 2000, 0, 500},
      {12345ull, 0ull, 64ull, 301, 0, 3},
      {12345ull, 0ull, 7ull, 53, 0, 2},
      // THE ONLY ROW THAT REACHES THE REJECTION LOOP FROM A REAL GENERATOR
      // WITH THE INDEX MAPPING IN PLACE. start = -2, end = INT_MAX gives
      // diff = 2^31 + 1, so t = 2^31 - 1 and about half of all draws are
      // rejected -- and it is also the only row whose returned value
      // overflows `int` (DEVIATION 186 in philox.mojo). A port that dropped
      // either the loop or the wrap is wrong here and nowhere else on device.
      {12345ull, 0ull, 512ull, 300, -2, 2147483647},
      // a non-zero base_subsequence, which cuML never uses but RngState
      // carries; see the note on `advance` in philox.mojo
      {12345ull, 110592ull, 256ull, 300, 0, 7},
    };
    for (const Case& c : cases) {
      std::vector<int> out(c.len, -1);
      uint32_t diff = (uint32_t)(c.end - c.start);
      fill_uniform_int(c.seed, c.base_sub, c.stride, out.data(), c.len, c.start, diff);
      printf("fill %llu %llu %llu %zu %d %d",
             (unsigned long long)c.seed,
             (unsigned long long)c.base_sub,
             (unsigned long long)c.stride,
             c.len, c.start, c.end);
      for (size_t i = 0; i < c.len; i++) printf(" %d", out[i]);
      printf("\n");
    }
  }

  // --- layer 6: cuML's actual call ---------------------------------------
  // `e2e <seed> <tree_id> <n_rows> <n_sampled> <stride> <rs> v0 .. v(n_sampled-1)`
  // Their seed chain and their argument shape: start = 0, end = n_rows,
  // len = n_sampled_rows. `n_rows = 1` and `n_rows = 2` are in here because a
  // one-row problem makes diff == 1, where the reduction returns `start`
  // unconditionally and a broken port looks perfect.
  // ---- raft::random::uniform<double> ---------------------------------
  // next_u64 pairs two next_u32 LOW WORD FIRST (rng_device.cuh:455-463);
  // next_double is (next_u64() >> 11) / 2^53 (:491-497); custom_next for
  // UniformDistParams multiplies by the SPAN and then adds the start
  // (:163-173). All three are separable failure modes, so all three are
  // printed. Doubles as decimal AND hex bits; the checker reads the hex.
  printf("section udbl\n");
  {
    struct DC { uint64_t seed; uint64_t sub; double start; double end; };
    const DC cases[] = {
      {12345ull, 0ull, 0.0, 1.0},
      {12345ull, 0ull, 0.0, 100.0},
      {12345ull, 7ull, 0.0, 1234.5},
      {0ull, 0ull, -5.0, 5.0},
      {16045690984503098046ull, 3ull, 0.0, 987654.321},
      {2615243109ull, 0ull, 0.0, 600.0},
    };
    for (const DC& c : cases) {
      PhiloxGenerator g(c.seed, c.sub, 0ull);
      printf("udbl %llu %llu %.17g %.17g",
             (unsigned long long)c.seed, (unsigned long long)c.sub,
             c.start, c.end);
      for (int i = 0; i < 8; i++) {
        // rng_device.cuh:455-463 then :491-497 then :163-173, verbatim
        uint32_t a = g.next_u32(), b = g.next_u32();
        uint64_t u = (uint64_t)a | ((uint64_t)b << 32);
        double res = (double)(u >> 11) / (double)(uint64_t(1) << 53);
        double v = (res * (c.end - c.start)) + c.start;
        uint64_t vb; memcpy(&vb, &v, 8);
        printf(" %.17g/%016llx", v, (unsigned long long)vb);
      }
      printf("\n");
    }
    // the raw u64 pairing, on its own
    {
      PhiloxGenerator g(12345ull, 0ull, 0ull);
      printf("u64 12345 0");
      for (int i = 0; i < 6; i++) {
        uint32_t a = g.next_u32(), b = g.next_u32();
        printf(" %llu",
               (unsigned long long)((uint64_t)a | ((uint64_t)b << 32)));
      }
      printf("\n");
    }
  }

  printf("section e2e\n");
  {
    struct Case {
      uint64_t seed;
      int tree_id;
      int n_rows;
      size_t n_sampled;
      uint64_t stride;
    };
    const Case cases[] = {
      {0ull, 0, 1000, 200, 110592ull},
      {0ull, 1, 1000, 200, 110592ull},
      {0ull, 2, 1000, 200, 110592ull},
      {12345ull, 0, 464809, 200, 110592ull},
      {12345ull, 7, 464809, 200, 110592ull},
      {16045690984503098046ull, 3, 581012, 200, 110592ull},
      // the high half of the seed is DISCARDED by their chain; these two
      // must produce identical rows, and the table proves it rather than
      // asserting it
      {4294967296ull, 3, 581012, 64, 110592ull},
      {0ull, 3, 581012, 64, 110592ull},
      {12345ull, 0, 1, 32, 110592ull},
      {12345ull, 0, 2, 32, 110592ull},
      // a bootstrap bigger than one grid: the loop runs
      {12345ull, 0, 1000, 700, 256ull},
    };
    for (const Case& c : cases) {
      uint32_t rs = rng_seed_for_tree(c.seed, c.tree_id);
      std::vector<int> out(c.n_sampled, -1);
      fill_uniform_int((uint64_t)rs, 0ull, c.stride, out.data(), c.n_sampled,
                       0, (uint32_t)c.n_rows);
      printf("e2e %llu %d %d %zu %llu %u",
             (unsigned long long)c.seed, c.tree_id, c.n_rows, c.n_sampled,
             (unsigned long long)c.stride, rs);
      for (size_t i = 0; i < c.n_sampled; i++) printf(" %d", out[i]);
      printf("\n");
    }
  }

  printf("end\n");
  return 0;
}
