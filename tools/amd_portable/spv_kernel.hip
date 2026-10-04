// Route B probe: what does ROCm's own amdgcnspirv flow emit, and does the
// HIP runtime on this box JIT it? Same math as k_pinned's fma part.
#include <hip/hip_runtime.h>
extern "C" __global__ void k_fma(float* o, const float* a, const float* b, const float* c, long n) {
  long i = blockIdx.x * 64 + threadIdx.x;
  if (i < n) o[i] = __builtin_fmaf(a[i], c[i], a[i] * b[i]);
}
