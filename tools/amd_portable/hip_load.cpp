// AMD portable-path probe 3: load a code object built OUTSIDE Mojo (from the
// IR Mojo emitted) with hipModuleLoadData, launch the probe kernels with the
// same integer-generated inputs as probe_native.mojo, print every output word.
// Usage: hip_load <kernel-label> <code-object> <symbol>   (one kernel per call)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <iterator>
#include <string>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "HIP %s: %s\n", #x, hipGetErrorString(e)); return 3; } } while (0)
static const int N = 4096, PER_LANE = 37;
static float gen(int i, uint32_t salt) {
  uint32_t u = ((uint32_t)i * 2654435761u) ^ salt;
  u ^= u >> 15; u *= 2246822519u; u ^= u >> 13;
  uint32_t e = 0x3E800000u + ((u >> 23) & 3u) * 0x00800000u;
  uint32_t bits = (u & 0x80000000u) | e | (u & 0x007FFFFFu);
  float f; memcpy(&f, &bits, 4); return f;
}
int main(int argc, char** argv) {
  if (argc != 4) { fprintf(stderr, "usage: hip_load label codeobj symbol\n"); return 2; }
  std::string label = argv[1];
  std::ifstream f(argv[2], std::ios::binary);
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});
  hipModule_t mod; hipFunction_t fn;
  CK(hipModuleLoadData(&mod, blob.data()));
  CK(hipModuleGetFunction(&fn, mod, argv[3]));
  int total = N * PER_LANE;
  std::vector<float> ha(total), hb(total), hc(total), ho(N);
  for (int i = 0; i < total; i++) { ha[i] = gen(i, 0x1234567u); hb[i] = gen(i, 0x89ABCDEu); hc[i] = gen(i, 0x5555AAAu); }
  float *a, *b, *c, *o;
  CK(hipMalloc(&a, total * 4)); CK(hipMalloc(&b, total * 4)); CK(hipMalloc(&c, total * 4)); CK(hipMalloc(&o, N * 4));
  CK(hipMemcpy(a, ha.data(), total * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(b, hb.data(), total * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(c, hc.data(), total * 4, hipMemcpyHostToDevice));
  int64_t n = N, pl = PER_LANE; int count = N;
  if (label == "k_dot") {
    void* args[] = {&o, &a, &b, &pl};
    CK(hipModuleLaunchKernel(fn, N / 64, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr));
    count = N / 64;
  } else {
    void* args[] = {&o, &a, &b, &c, &n};
    CK(hipModuleLaunchKernel(fn, N / 64, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr));
  }
  CK(hipDeviceSynchronize());
  CK(hipMemcpy(ho.data(), o, count * 4, hipMemcpyDeviceToHost));
  printf("=== %s\n", label.c_str());
  for (int i = 0; i < count; i++) { uint32_t w; memcpy(&w, &ho[i], 4); printf("0x%x\n", w); }
  return 0;
}
