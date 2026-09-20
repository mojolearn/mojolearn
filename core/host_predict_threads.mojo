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
"""
from std.os import getenv
from std.sys.info import num_physical_cores

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
