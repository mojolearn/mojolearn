// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Andrew Hendel. Part of mojolearn
//
// tools/ops/call_profile_malloc.c -- the allocation counter of tools/ops/call_profile_job.sh (diagnostic tooling, never
// product). Built on the pod (`cc -O2 -shared -fPIC -o callprof_malloc.so call_profile_malloc.c`) and LD_PRELOADed
// into the profiled race: it counts the libc heap calls of every thread (malloc, calloc, realloc, free, the aligned
// family) and the bytes requested, forwarding each to glibc's own entry points (__libc_*: no dlsym, so no bootstrap
// allocation). tools/bench_board_small_rows.py reads `callprof_counts` at the open and the close of the profile window;
// the difference over the window's calls is the allocations per call. Relaxed atomic adds: a few nanoseconds per
// call, run only in the strace pass (already perturbed), never in a timing pass. Allocators that bypass libc (a
// runtime's own mmap arena) show in the strace pass's mmap/munmap counts instead.
#define _GNU_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <errno.h>

extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void __libc_free(void *);
extern void *__libc_memalign(size_t, size_t);

// the order tools/bench_board_small_rows.py MALLOC_FIELDS names
enum { C_MALLOC, C_CALLOC, C_REALLOC, C_FREE, C_ALIGNED, C_BYTES, C_N };
static uint64_t counts[C_N];

static inline void bump(int k, uint64_t bytes) {
    __atomic_fetch_add(&counts[k], 1, __ATOMIC_RELAXED);
    if (bytes) __atomic_fetch_add(&counts[C_BYTES], bytes, __ATOMIC_RELAXED);
}

void callprof_counts(uint64_t *out, int n) {
    for (int i = 0; i < n && i < C_N; i++) out[i] = __atomic_load_n(&counts[i], __ATOMIC_RELAXED);
}

void *malloc(size_t n) { bump(C_MALLOC, n); return __libc_malloc(n); }
void *calloc(size_t a, size_t b) { bump(C_CALLOC, a * b); return __libc_calloc(a, b); }
void *realloc(void *p, size_t n) { bump(C_REALLOC, n); return __libc_realloc(p, n); }
void free(void *p) { if (p) bump(C_FREE, 0); __libc_free(p); }
void *memalign(size_t al, size_t n) { bump(C_ALIGNED, n); return __libc_memalign(al, n); }
void *aligned_alloc(size_t al, size_t n) { bump(C_ALIGNED, n); return __libc_memalign(al, n); }
void *valloc(size_t n) { bump(C_ALIGNED, n); return __libc_memalign(4096, n); }
int posix_memalign(void **out, size_t al, size_t n) {
    if (al % sizeof(void *) != 0 || (al & (al - 1)) != 0) return EINVAL;
    bump(C_ALIGNED, n);
    void *p = __libc_memalign(al, n);
    if (!p && n) return ENOMEM;
    *out = p;
    return 0;
}
