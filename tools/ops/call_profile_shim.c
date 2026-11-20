// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Andrew Hendel. Part of mojolearn
//
// tools/ops/call_profile_shim.c -- the LD_PRELOAD timing shim of tools/ops/call_profile_job.sh's `shim` pass
// (diagnostic tooling, never product). Built on the pod (`cc -O2 -fno-builtin -shared -fPIC -o callprof_shim.so
// call_profile_shim.c -ldl`). No ptrace, no perf events, no root: it counts and times, per entry point, every call
// of the process's threads to
//   - the CUDA driver entries the nsys pass counts: cuLaunchKernel(Ex), cuStreamSynchronize, cuCtxSynchronize,
//     cuStreamIsCapturing, cuMemcpy* (the _v2 and async forms), cuMemAlloc* and cuMemFree*;
//   - libc: malloc, calloc, realloc, free, the aligned family; mmap and munmap; memcpy and memset of 64 KiB or more.
// The Mojo runtime opens libcuda with dlopen and looks each entry up with dlsym (no direct link), so the shim also
// wraps dlsym (and cuGetProcAddress): a lookup of a wrapped entry returns the wrapper and keeps the real pointer.
// tools/bench_board_small_rows.py reads `callprof_shim_read` at the open and the close of the profile window; the
// difference over the window's calls is the count, nanoseconds and bytes per call of each entry.
// Cost: two CLOCK_MONOTONIC reads (vDSO) and three relaxed atomic adds per wrapped call; memcpy/memset under
// 64 KiB pass straight through. A dlsym(RTLD_NEXT, ...) from another library resolves relative to this shim (the
// first object after the executable), the one deviation from glibc's semantics.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void __libc_free(void *);
extern void *__libc_memalign(size_t, size_t);
extern void *__memset_chk(void *, int, size_t, size_t);

#define BIG_COPY ((size_t)65536)

// X(name, arity, index of the byte-count argument or -1); the order is the report's
#define CUDA_ENTRIES(X)                         \
    X(cuLaunchKernelEx, 4, -1)                  \
    X(cuLaunchKernel, 11, -1)                   \
    X(cuStreamSynchronize, 1, -1)               \
    X(cuCtxSynchronize, 0, -1)                  \
    X(cuStreamIsCapturing, 2, -1)               \
    X(cuMemcpyHtoDAsync_v2, 4, 2)               \
    X(cuMemcpyDtoHAsync_v2, 4, 2)               \
    X(cuMemcpyDtoDAsync_v2, 4, 2)               \
    X(cuMemcpyHtoD_v2, 3, 2)                    \
    X(cuMemcpyDtoH_v2, 3, 2)                    \
    X(cuMemcpyDtoD_v2, 3, 2)                    \
    X(cuMemcpyAsync, 4, 2)                      \
    X(cuMemcpy, 3, 2)                           \
    X(cuMemAlloc_v2, 2, 1)                      \
    X(cuMemAllocAsync, 3, 1)                    \
    X(cuMemAllocFromPoolAsync, 4, 1)            \
    X(cuMemAllocHost_v2, 2, 1)                  \
    X(cuMemAllocManaged, 3, 1)                  \
    X(cuMemFree_v2, 1, -1)                      \
    X(cuMemFreeAsync, 2, -1)                    \
    X(cuMemFreeHost, 1, -1)

#define LIBC_ENTRIES(X) X(malloc) X(calloc) X(realloc) X(free) X(aligned) X(mmap) X(munmap) X(memcpy_64k) X(memset_64k)

enum {
#define E_CUDA(n, a, b) E_##n,
#define E_LIBC(n) E_##n,
    CUDA_ENTRIES(E_CUDA) LIBC_ENTRIES(E_LIBC) E_N
#undef E_CUDA
#undef E_LIBC
};

static const char *const NAMES[E_N] = {
#define N_CUDA(n, a, b) #n,
#define N_LIBC(n) #n,
    CUDA_ENTRIES(N_CUDA) LIBC_ENTRIES(N_LIBC)
#undef N_CUDA
#undef N_LIBC
};

static uint64_t counts[E_N][3];     // calls, nanoseconds, bytes
// > 0 while this thread is inside a wrapped CUDA entry: its libc calls (the driver's staging copies, its allocations)
// are part of that entry's time and are not counted again under libc (the report adds the two)
static __thread int in_cuda __attribute__((tls_model("initial-exec")));

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline void add(int k, uint64_t t0, uint64_t bytes) {
    if (k >= E_malloc && in_cuda) return;
    uint64_t dt = now_ns() - t0;
    __atomic_fetch_add(&counts[k][0], 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&counts[k][1], dt, __ATOMIC_RELAXED);
    if (bytes) __atomic_fetch_add(&counts[k][2], bytes, __ATOMIC_RELAXED);
}

// ---- the read side (ctypes) ---------------------------------------------------------------------------------------
int callprof_shim_n(void) { return E_N; }
const char *callprof_shim_name(int i) { return (i >= 0 && i < E_N) ? NAMES[i] : ""; }
void callprof_shim_read(uint64_t *out, int n) {     // n entries of (calls, ns, bytes)
    for (int i = 0; i < n && i < E_N; i++)
        for (int j = 0; j < 3; j++) out[3 * i + j] = __atomic_load_n(&counts[i][j], __ATOMIC_RELAXED);
}

// ---- dlsym: the real lookup, versioned (glibc >= 2.34 keeps it in libc, older in libdl) -------------------------
typedef void *(*dlsym_fn)(void *, const char *);
static dlsym_fn real_dlsym_fn;
static dlsym_fn get_real_dlsym(void) {
    if (!real_dlsym_fn) {
        static const char *const vers[] = {"GLIBC_2.34", "GLIBC_2.17", "GLIBC_2.2.5", "GLIBC_2.0"};
        for (unsigned i = 0; i < sizeof(vers) / sizeof(vers[0]) && !real_dlsym_fn; i++)
            real_dlsym_fn = (dlsym_fn)dlvsym(RTLD_NEXT, "dlsym", vers[i]);
    }
    return real_dlsym_fn;
}

// ---- the CUDA wrappers: every argument is an integer or a pointer (CUresult is an int) ----------------------------
typedef uintptr_t U;
#define ARGS0 void
#define ARGS1 U a0
#define ARGS2 U a0, U a1
#define ARGS3 U a0, U a1, U a2
#define ARGS4 U a0, U a1, U a2, U a3
#define ARGS11 U a0, U a1, U a2, U a3, U a4, U a5, U a6, U a7, U a8, U a9, U a10
#define PASS0
#define PASS1 a0
#define PASS2 a0, a1
#define PASS3 a0, a1, a2
#define PASS4 a0, a1, a2, a3
#define PASS11 a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10

#define DEF_CUDA(n, ar, bi)                                                                     \
    typedef int (*n##_fn)(ARGS##ar);                                                            \
    static n##_fn real_##n;                                                                     \
    int n(ARGS##ar);
CUDA_ENTRIES(DEF_CUDA)
#undef DEF_CUDA

static void *resolve_next(const char *name) {
    dlsym_fn d = get_real_dlsym();
    void *p = d ? d(RTLD_NEXT, name) : NULL;
    if (!p) {       // libcuda opened later with RTLD_LOCAL: look in it by name
        void *h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (h && d) p = d(h, name);
    }
    return p;
}

// the byte count of argument `bi` (0-based) of a call, 0 when the entry has none
#define BYTES_OF(bi, ...) ((bi) < 0 ? 0 : ((const U[]){__VA_ARGS__, 0})[(bi) < 0 ? 0 : (bi)])
#define BODY(n, ar, bi, ...)                                                                    \
    if (!real_##n) real_##n = (n##_fn)resolve_next(#n);                                         \
    if (!real_##n) return 999; /* CUDA_ERROR_UNKNOWN */                                         \
    uint64_t t0 = now_ns();                                                                     \
    in_cuda++;                                                                                  \
    int r = real_##n(PASS##ar);                                                                 \
    in_cuda--;                                                                                  \
    add(E_##n, t0, (uint64_t)BYTES_OF(bi, __VA_ARGS__));                                        \
    return r;

int cuLaunchKernelEx(ARGS4) { BODY(cuLaunchKernelEx, 4, -1, PASS4) }
int cuLaunchKernel(ARGS11) { BODY(cuLaunchKernel, 11, -1, PASS11) }
int cuStreamSynchronize(ARGS1) { BODY(cuStreamSynchronize, 1, -1, PASS1) }
int cuCtxSynchronize(void) { BODY(cuCtxSynchronize, 0, -1, 0) }
int cuStreamIsCapturing(ARGS2) { BODY(cuStreamIsCapturing, 2, -1, PASS2) }
int cuMemcpyHtoDAsync_v2(ARGS4) { BODY(cuMemcpyHtoDAsync_v2, 4, 2, PASS4) }
int cuMemcpyDtoHAsync_v2(ARGS4) { BODY(cuMemcpyDtoHAsync_v2, 4, 2, PASS4) }
int cuMemcpyDtoDAsync_v2(ARGS4) { BODY(cuMemcpyDtoDAsync_v2, 4, 2, PASS4) }
int cuMemcpyHtoD_v2(ARGS3) { BODY(cuMemcpyHtoD_v2, 3, 2, PASS3) }
int cuMemcpyDtoH_v2(ARGS3) { BODY(cuMemcpyDtoH_v2, 3, 2, PASS3) }
int cuMemcpyDtoD_v2(ARGS3) { BODY(cuMemcpyDtoD_v2, 3, 2, PASS3) }
int cuMemcpyAsync(ARGS4) { BODY(cuMemcpyAsync, 4, 2, PASS4) }
int cuMemcpy(ARGS3) { BODY(cuMemcpy, 3, 2, PASS3) }
int cuMemAlloc_v2(ARGS2) { BODY(cuMemAlloc_v2, 2, 1, PASS2) }
int cuMemAllocAsync(ARGS3) { BODY(cuMemAllocAsync, 3, 1, PASS3) }
int cuMemAllocFromPoolAsync(ARGS4) { BODY(cuMemAllocFromPoolAsync, 4, 1, PASS4) }
int cuMemAllocHost_v2(ARGS2) { BODY(cuMemAllocHost_v2, 2, 1, PASS2) }
int cuMemAllocManaged(ARGS3) { BODY(cuMemAllocManaged, 3, 1, PASS3) }
int cuMemFree_v2(ARGS1) { BODY(cuMemFree_v2, 1, -1, PASS1) }
int cuMemFreeAsync(ARGS2) { BODY(cuMemFreeAsync, 2, -1, PASS2) }
int cuMemFreeHost(ARGS1) { BODY(cuMemFreeHost, 1, -1, PASS1) }

// a looked-up entry the shim wraps: keep the real pointer, hand back the wrapper (NULL: not wrapped)
static void *hook(const char *name, void *real) {
    if (!name || !real || name[0] != 'c' || name[1] != 'u') return NULL;
#define HOOK(n, ar, bi)                                                                         \
    if (strcmp(name, #n) == 0) {                                                                \
        if (!real_##n && real != (void *)&n) real_##n = (n##_fn)real;                           \
        return (void *)&n;                                                                      \
    }
    CUDA_ENTRIES(HOOK)
#undef HOOK
    return NULL;
}

void *dlsym(void *handle, const char *name) {
    dlsym_fn d = get_real_dlsym();
    void *r = d ? d(handle, name) : NULL;
    void *w = hook(name, r);
    return w ? w : r;
}

// cuGetProcAddress(symbol, pfn, cudaVersion, flags[, status]): the base name (no _v2) is asked for; map the forms
// the shim wraps (the driver hands back its newest version of the base name, the _v2 one for the copies/allocs)
typedef int (*gpa_fn)(const char *, void **, int, uint64_t);
typedef int (*gpa2_fn)(const char *, void **, int, uint64_t, void *);
static gpa_fn real_gpa;
static gpa2_fn real_gpa2;
static void hook_gpa(const char *sym, void **pfn) {
    if (!sym || !pfn || !*pfn) return;
    char buf[96];
    void *w = hook(sym, *pfn);
    if (!w && strlen(sym) + 4 < sizeof(buf)) {
        strcpy(buf, sym);
        strcat(buf, "_v2");
        w = hook(buf, *pfn);
    }
    if (w) *pfn = w;
}
int cuGetProcAddress(const char *sym, void **pfn, int ver, uint64_t flags) {
    if (!real_gpa) real_gpa = (gpa_fn)resolve_next("cuGetProcAddress");
    if (!real_gpa) return 999;
    int r = real_gpa(sym, pfn, ver, flags);
    if (r == 0) hook_gpa(sym, pfn);
    return r;
}
int cuGetProcAddress_v2(const char *sym, void **pfn, int ver, uint64_t flags, void *status) {
    if (!real_gpa2) real_gpa2 = (gpa2_fn)resolve_next("cuGetProcAddress_v2");
    if (!real_gpa2) return 999;
    int r = real_gpa2(sym, pfn, ver, flags, status);
    if (r == 0) hook_gpa(sym, pfn);
    return r;
}

// ---- libc --------------------------------------------------------------------------------------------------------
void *malloc(size_t n) { uint64_t t = now_ns(); void *p = __libc_malloc(n); add(E_malloc, t, n); return p; }
void *calloc(size_t a, size_t b) { uint64_t t = now_ns(); void *p = __libc_calloc(a, b); add(E_calloc, t, a * b); return p; }
void *realloc(void *q, size_t n) { uint64_t t = now_ns(); void *p = __libc_realloc(q, n); add(E_realloc, t, n); return p; }
void free(void *p) { if (!p) return; uint64_t t = now_ns(); __libc_free(p); add(E_free, t, 0); }
void *memalign(size_t al, size_t n) { uint64_t t = now_ns(); void *p = __libc_memalign(al, n); add(E_aligned, t, n); return p; }
void *aligned_alloc(size_t al, size_t n) { return memalign(al, n); }
void *valloc(size_t n) { return memalign(4096, n); }
int posix_memalign(void **out, size_t al, size_t n) {
    if (al % sizeof(void *) != 0 || (al & (al - 1)) != 0) return EINVAL;
    void *p = memalign(al, n);
    if (!p && n) return ENOMEM;
    *out = p;
    return 0;
}
void *mmap(void *a, size_t n, int prot, int fl, int fd, off_t off) {
    uint64_t t = now_ns();
    void *p = (void *)syscall(SYS_mmap, a, n, prot, fl, fd, off);
    add(E_mmap, t, n);
    return p;
}
void *mmap64(void *a, size_t n, int prot, int fl, int fd, off_t off) { return mmap(a, n, prot, fl, fd, off); }
int munmap(void *a, size_t n) {
    uint64_t t = now_ns();
    int r = (int)syscall(SYS_munmap, a, n);
    add(E_munmap, t, n);
    return r;
}
// memmove and __memset_chk are libc's own exports (never interposed here): no lookup, no recursion
void *memcpy(void *d, const void *s, size_t n) {
    if (n < BIG_COPY) return memmove(d, s, n);
    uint64_t t = now_ns();
    void *r = memmove(d, s, n);
    add(E_memcpy_64k, t, n);
    return r;
}
void *memset(void *d, int c, size_t n) {
    if (n < BIG_COPY) return __memset_chk(d, c, n, (size_t)-1);
    uint64_t t = now_ns();
    void *r = __memset_chk(d, c, n, (size_t)-1);
    add(E_memset_64k, t, n);
    return r;
}
