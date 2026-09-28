# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host tasks' floating-point environment. HOST ONLY
(lane/algos-linear, 2026-09-27).

MAX's CPU worker threads run with MXCSR FTZ and DAZ set (measured on an
x86 RunPod box: pool threads 0x9fe0, the calling thread 0x1fa0), while
the calling thread keeps the IEEE default. A task that produces a
subnormal therefore flushed it to zero on a pool thread and kept it on
the calling thread, so the bits depended on the task count: the float64
`predict_proba` link (`p = 1/(1 + exp(709.39))` = 8.2e-309) read 0 on
the CPU column's pool and 8.2e-309 on the GPU binding's serial host
link (lane logistic-unpenalized-no-intercept/dupes, infer and batch
DIVERGENT at MOJOLEARN_CPU_THREADS unset, IDENTICAL at 1). Every task of
a row split therefore runs between `host_ieee_fp_enter()` and
`host_ieee_fp_leave()`: FTZ and DAZ cleared (x86 MXCSR bits 15 and 6;
Arm FPCR.FZ, bit 24) for the task's arithmetic, the pool's own word
restored after it. Float32 paths flush by bits (IDENTITY_PATHS row 10),
so the IEEE environment moves none of their results.

Callers today: the row splits of `core/classical_host_predict.mojo` and
`glm/estimator.mojo::qn_softmax_host` (the rows of `core/host_predict_threads.
mojo`'s policy). Every other host `sync_parallelize` site runs on the same
pool and is owed the same two calls by the lane that owns it.
"""
from std.sys import CompilationTarget, llvm_intrinsic
from std.memory import stack_allocation

#: MXCSR's flush-to-zero (bit 15) and denormals-are-zero (bit 6) bits.
comptime _MXCSR_FTZ_DAZ = UInt32(0x8040)
#: FPCR.FZ, Arm's flush-to-zero bit.
comptime _FPCR_FZ = UInt64(1) << 24


@always_inline
def host_ieee_fp_enter() -> UInt64:
    """Clear flush-to-zero and denormals-are-zero on the CURRENT thread and
    return the word it had, for `host_ieee_fp_leave`. See the module note:
    a MAX pool thread starts with both set. The environment write is an
    intrinsic with side effects on memory, so the task's loads (and the
    arithmetic that reads them) stay after it."""
    comptime if CompilationTarget.is_x86():
        var cell = stack_allocation[1, UInt32]()
        llvm_intrinsic["llvm.x86.sse.stmxcsr", NoneType](cell)
        var saved = cell.load()
        cell.store(saved & ~_MXCSR_FTZ_DAZ)
        llvm_intrinsic["llvm.x86.sse.ldmxcsr", NoneType](cell)
        return UInt64(saved)
    elif CompilationTarget.has_neon():
        var saved = llvm_intrinsic["llvm.aarch64.get.fpcr", UInt64]()
        llvm_intrinsic["llvm.aarch64.set.fpcr", NoneType](saved & ~_FPCR_FZ)
        return saved
    else:
        return UInt64(0)


@always_inline
def host_ieee_fp_leave(saved: UInt64):
    """Restore the word `host_ieee_fp_enter` returned (the task's stores are
    memory writes, so they stay before it)."""
    comptime if CompilationTarget.is_x86():
        var cell = stack_allocation[1, UInt32]()
        cell.store(UInt32(saved))
        llvm_intrinsic["llvm.x86.sse.ldmxcsr", NoneType](cell)
    elif CompilationTarget.has_neon():
        llvm_intrinsic["llvm.aarch64.set.fpcr", NoneType](saved)
