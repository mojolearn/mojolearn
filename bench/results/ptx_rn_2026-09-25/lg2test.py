#!/usr/bin/env python3
"""The gbdt fold-stripe expression exactly as 0.8.19's IDENTICAL PTX spells it
(gbdt_methods_kernel_pointwise: cvt.rn.f32.u32; lg2.approx.ftz.f32; cvt.rpi;
cvt.rzi.u32), run on this GPU for n = 1 .. 2^24 and compared with the exact
integer ceil(log2(n)). Also sqrt.approx.ftz.f32 against sqrt.rn.f32 over every
positive normal float32, counted (the fused k-NN arm's spelling)."""
import ctypes, json, sys
import numpy as np

N = 1 << 24
PTX = r"""
.version 8.0
.target %s
.address_size 64
.visible .entry lg2k(.param .u64 out, .param .u32 n)
{
  .reg .pred %%p<2>; .reg .b32 %%r<12>; .reg .f32 %%f<4>; .reg .b64 %%rd<4>;
  mov.u32 %%r1, %%ctaid.x; mov.u32 %%r2, %%ntid.x; mov.u32 %%r3, %%tid.x;
  mad.lo.s32 %%r4, %%r1, %%r2, %%r3;
  ld.param.u32 %%r5, [n];
  setp.ge.u32 %%p1, %%r4, %%r5;
  @%%p1 bra DONE;
  add.s32 %%r6, %%r4, 1;
  cvt.rn.f32.u32 %%f1, %%r6;
  lg2.approx.ftz.f32 %%f2, %%f1;
  cvt.rpi.f32.f32 %%f3, %%f2;
  cvt.rzi.u32.f32 %%r7, %%f3;
  ld.param.u64 %%rd1, [out];
  mul.wide.u32 %%rd2, %%r4, 4;
  add.s64 %%rd3, %%rd1, %%rd2;
  st.global.u32 [%%rd3], %%r7;
DONE:
  ret;
}
.visible .entry sqk(.param .u64 out, .param .u32 base, .param .u32 n)
{
  .reg .pred %%p<2>; .reg .b32 %%r<12>; .reg .f32 %%f<4>; .reg .b64 %%rd<4>;
  mov.u32 %%r1, %%ctaid.x; mov.u32 %%r2, %%ntid.x; mov.u32 %%r3, %%tid.x;
  mad.lo.s32 %%r4, %%r1, %%r2, %%r3;
  ld.param.u32 %%r5, [n];
  setp.ge.u32 %%p1, %%r4, %%r5;
  @%%p1 bra DONE2;
  ld.param.u32 %%r6, [base];
  add.s32 %%r7, %%r6, %%r4;
  mov.b32 %%f1, %%r7;
  sqrt.approx.ftz.f32 %%f2, %%f1;
  sqrt.rn.f32 %%f3, %%f1;
  mov.b32 %%r8, %%f2; mov.b32 %%r9, %%f3;
  setp.ne.u32 %%p1, %%r8, %%r9;
  @!%%p1 bra DONE2;
  ld.param.u64 %%rd1, [out];
  atom.global.add.u32 %%r10, [%%rd1], 1;
DONE2:
  ret;
}
"""
cu = ctypes.CDLL('libcuda.so.1'); cu.cuInit(0)
dev = ctypes.c_int(); cu.cuDeviceGet(ctypes.byref(dev), 0)
maj, mnr = ctypes.c_int(), ctypes.c_int()
cu.cuDeviceGetAttribute(ctypes.byref(maj), 75, dev); cu.cuDeviceGetAttribute(ctypes.byref(mnr), 76, dev)
sm = 'sm_%d%d' % (maj.value, mnr.value)
ctx = ctypes.c_void_p(); cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev); cu.cuCtxSetCurrent(ctx)
mod = ctypes.c_void_p()
src = (PTX % sm).encode() + b'\0'
assert cu.cuModuleLoadData(ctypes.byref(mod), src) == 0, 'module load'
f = ctypes.c_void_p(); assert cu.cuModuleGetFunction(ctypes.byref(f), mod, b'lg2k') == 0
g = ctypes.c_void_p(); assert cu.cuModuleGetFunction(ctypes.byref(g), mod, b'sqk') == 0
buf = ctypes.c_uint64(); assert cu.cuMemAlloc_v2(ctypes.byref(buf), ctypes.c_size_t(N * 4)) == 0
p_out = ctypes.c_uint64(buf.value); p_n = ctypes.c_uint32(N)
args = (ctypes.c_void_p * 2)(ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p), ctypes.cast(ctypes.byref(p_n), ctypes.c_void_p))
assert cu.cuLaunchKernel(f, (N + 255) // 256, 1, 1, 256, 1, 1, 0, None, args, None) == 0
assert cu.cuCtxSynchronize() == 0
host = np.empty(N, dtype=np.uint32)
assert cu.cuMemcpyDtoH_v2(host.ctypes.data_as(ctypes.c_void_p), buf, ctypes.c_size_t(N * 4)) == 0
n = np.arange(1, N + 1, dtype=np.int64)
exact = np.array([(int(v) - 1).bit_length() for v in (1, 2, 3)], dtype=np.int64)  # sanity: 0,1,2
assert list(exact) == [0, 1, 2]
exact = np.ceil(np.log2(n.astype(np.float64))).astype(np.int64)   # exact for n <= 2^24
bad = np.nonzero(host.astype(np.int64) != exact)[0]
# sqrt: every positive normal float32 (bits 0x00800000 .. 0x7f7fffff), in chunks
cnt = ctypes.c_uint64(); assert cu.cuMemAlloc_v2(ctypes.byref(cnt), ctypes.c_size_t(4)) == 0
assert cu.cuMemsetD32_v2(cnt, 0, ctypes.c_size_t(1)) == 0
lo, hi, chunk = 0x00800000, 0x7F800000, 1 << 26
b = lo
while b < hi:
    m = min(chunk, hi - b)
    p_c = ctypes.c_uint64(cnt.value); p_b = ctypes.c_uint32(b); p_m = ctypes.c_uint32(m)
    a2 = (ctypes.c_void_p * 3)(*(ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in (p_c, p_b, p_m)))
    assert cu.cuLaunchKernel(g, (m + 255) // 256, 1, 1, 256, 1, 1, 0, None, a2, None) == 0
    b += m
assert cu.cuCtxSynchronize() == 0
nsq = ctypes.c_uint32(); assert cu.cuMemcpyDtoH_v2(ctypes.byref(nsq), cnt, ctypes.c_size_t(4)) == 0
res = {'sm': sm, 'lg2_n_tested': N, 'lg2_ceil_mismatches': int(bad.size),
       'lg2_first_mismatches': [[int(n[i]), int(host[i]), int(exact[i])] for i in bad[:10]],
       'sqrt_approx_ne_rn_over_positive_normals': int(nsq.value), 'positive_normals': hi - lo}
print(json.dumps(res))
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else 'lg2test.json', 'w'), indent=1)
