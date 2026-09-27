# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The row split of the CPU inference paths (lane/infer-speed-classical,
2026-09-17, DEVIATION 2920).

HOST ONLY. The classical, k-NN and KDE host inference entries
(`core/classical_host_predict.mojo`, `core/knn_host_predict.mojo`,
`kde/host/kde_oracle.mojo`) used to walk every output row on the calling
thread. Each output row of those paths is a function of its own input row
and the fitted state alone: a gemm cell reads row `i` of X and row `j` of
the fitted matrix, a k-NN row reads query `i` against the whole index, a
KDE score reads query `i` against the whole training set. No fold crosses a
row, so rows may run on different threads with every statement of a row
kept in its order, and the bits are the bits of the serial walk. This file
holds the one policy every such path reads, so a thread count is one
sentence in one place:

  MOJOLEARN_CPU_THREADS   when set to a positive integer, the ceiling on
                          the tasks one call splits its rows into; the
                          tooling that pins a box to one core exports it as
                          1 (tools/mac_slot.py, tools/nvidia_serial_guard.py),
                          and that setting makes every path here serial.
  unset or not positive   one task per physical core (`num_physical_cores`),
                          the byte LM host's default (`training/byte_lm_host.
                          mojo::byte_host_worker_count`).

A task owns a CONTIGUOUS range of rows, `[c * chunk, min((c + 1) * chunk,
rows))`, so `tasks` never exceeds `rows` and a single task runs on the
calling thread without touching the pool. Tasks share nothing they write:
each writes its own rows of the caller's output and reads the inputs only.
No task starts another parallel region, and an owner a task reads through a
pointer is transferred only after the join (the step-33 race class,
`gbdt/train.mojo`).

THIS IS NOT A NUMERIC ROW. It changes which thread computes a row, never
what the row computes, so the kernel matrix does not carry it; the CPU
identity column proves it by reading every cell IDENTICAL at the default
count and at 1.

THE POOL'S FLOATING-POINT ENVIRONMENT (lane/algos-linear, 2026-09-27).
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
"""
from std.os import getenv
from std.sys import CompilationTarget, llvm_intrinsic
from std.sys.info import num_physical_cores
from std.memory import stack_allocation

#: The most tasks one call splits into, whatever the box reports; the byte
#: LM host's `BYTE_HOST_MAX_THREADS` bound, kept so a misread environment
#: cannot ask the pool for a million tasks.
comptime HOST_PREDICT_MAX_TASKS = 1024

#: The environment name, spelled once.
comptime HOST_PREDICT_THREADS_ENV = "MOJOLEARN_CPU_THREADS"


def host_predict_threads_requested() -> Int:
    """The ceiling MOJOLEARN_CPU_THREADS asks for, or 0 when it is unset,
    empty, not an integer or not positive (each of those means "the
    default"; a misspelled setting is never a silent one-thread run and
    never a refusal, because a thread count moves no bit)."""
    var s = String(getenv(HOST_PREDICT_THREADS_ENV))
    if s == "":
        return 0
    try:
        var v = Int(atol(s))
        if v > 0:
            return v
        return 0
    except:
        return 0


def host_predict_task_count(rows: Int) -> Int:
    """How many contiguous row tasks a call over `rows` output rows splits
    into: the environment's ceiling when it names one, else one per
    physical core, never more than `rows`, never more than
    HOST_PREDICT_MAX_TASKS, at least 1."""
    if rows <= 1:
        return 1
    var workers = host_predict_threads_requested()
    if workers <= 0:
        workers = num_physical_cores()
    if workers < 1:
        workers = 1
    if workers > HOST_PREDICT_MAX_TASKS:
        workers = HOST_PREDICT_MAX_TASKS
    if workers > rows:
        workers = rows
    return workers


@always_inline
def host_predict_chunk(rows: Int, tasks: Int) -> Int:
    """Rows per task, the ceiling of `rows / tasks`; task `c` owns
    `[c * chunk, min((c + 1) * chunk, rows))`."""
    if tasks < 1:
        return rows
    return (rows + tasks - 1) // tasks


#: The pointer every `_into` entry takes: the host bindings' own type
#: (`bindings/hostptr.mojo::f32_ptr`), so a binding hands the caller's
#: address straight through and a List door rebinds its storage to it.
comptime HostF32Ptr = MutPointer[Float32, MutUntrackedOrigin]
comptime HostF64Ptr = MutPointer[Float64, MutUntrackedOrigin]
comptime HostU32Ptr = MutPointer[UInt32, MutUntrackedOrigin]
comptime HostI64Ptr = MutPointer[Int64, MutUntrackedOrigin]


@always_inline
def host_list_ptr(x: List[Float32]) -> HostF32Ptr:
    """A List's storage as `HostF32Ptr`, the `rebind` `core/gram_multi_gpu.
    mojo` performs on its shard list. The List must outlive every read
    and write through the result; the callers here keep it in a local
    until after the join."""
    return rebind[HostF32Ptr](x.unsafe_ptr())


@always_inline
def host_list_ptr_u32(x: List[UInt32]) -> HostU32Ptr:
    """`host_list_ptr` for the k-NN index output."""
    return rebind[HostU32Ptr](x.unsafe_ptr())


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
