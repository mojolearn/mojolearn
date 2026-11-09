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
from core.shared_slot import SharedSlot

#: The most tasks one call splits into, whatever the box reports; the byte
#: LM host's `BYTE_HOST_MAX_THREADS` bound, kept so a misread environment
#: cannot ask the pool for a million tasks.
comptime HOST_PREDICT_MAX_TASKS = 1024

#: The environment name, spelled once.
comptime HOST_PREDICT_THREADS_ENV = "MOJOLEARN_CPU_THREADS"


#: Lane perf-host-grain (2026-11-06): the smallest amount of work, in scalar
#: element operations (one load plus one arithmetic step on one element), that
#: one host task is given. Cost reasoning, not a shape: a `sync_parallelize`
#: region enqueues its tasks, wakes parked pool workers (a futex wake each)
#: and joins on a latch, which costs on the order of 5-50 microseconds before
#: the first useful instruction, more when the workers are asleep. 2^16
#: scalar operations at about 1 ns each is 30-70 microseconds, so a task this
#: size is at least as large as the dispatch it pays for; below twice this
#: amount a split cannot give two tasks a worthwhile share and the call runs
#: on the calling thread. `core/host_lanes.mojo`'s HOST_ROW_TASK_MIN_WORK was
#: already this value and now reads it from here.
comptime HOST_MIN_TASK_WORK = 1 << 16


def _host_threads_env() -> Int:
    """MOJOLEARN_CPU_THREADS as a positive integer, or 0 when it is unset,
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


def _host_threads_resolve(requested: Int) -> Int:
    """The worker ceiling for a request: the request when positive, else
    one per physical core, clamped to [1, HOST_PREDICT_MAX_TASKS]."""
    var workers = requested
    if workers <= 0:
        workers = num_physical_cores()
    if workers < 1:
        workers = 1
    if workers > HOST_PREDICT_MAX_TASKS:
        workers = HOST_PREDICT_MAX_TASKS
    return workers


struct _HostThreadPolicy(Defaultable, Movable):
    """The thread policy, read ONCE per process (lane perf-host-grain,
    2026-11-06): every split used to read the environment, build a String
    and parse it, and ask the system for the core count. The process-wide
    named global is shared by every host binding loaded in the process, so
    all of them see one value. A change of MOJOLEARN_CPU_THREADS after the
    first split of the process is not seen; set it before importing."""
    var requested: Int
    var workers: Int

    def __init__(out self):
        self.requested = _host_threads_env()
        self.workers = _host_threads_resolve(self.requested)


comptime _HOST_THREAD_POLICY = SharedSlot[_HostThreadPolicy, "MojolearnHostThreadPolicyV2"]


def host_predict_threads_requested() -> Int:
    """The ceiling MOJOLEARN_CPU_THREADS asks for, or 0 when it is unset,
    empty, not an integer or not positive. Read once per process."""
    try:
        return _HOST_THREAD_POLICY.get_or_create_ptr()[].requested
    except:
        return _host_threads_env()


def host_predict_workers() -> Int:
    """The resolved worker ceiling (environment, else physical cores,
    clamped to [1, HOST_PREDICT_MAX_TASKS]). Read once per process."""
    try:
        return _HOST_THREAD_POLICY.get_or_create_ptr()[].workers
    except:
        return _host_threads_resolve(_host_threads_env())


def host_predict_task_count(rows: Int) -> Int:
    """How many contiguous row tasks a call over `rows` output rows splits
    into: the environment's ceiling when it names one, else one per
    physical core, never more than `rows`, never more than
    HOST_PREDICT_MAX_TASKS, at least 1. For a call whose work per row is
    known, `host_predict_task_count_work` also applies the grain floor."""
    if rows <= 1:
        return 1
    var workers = host_predict_workers()
    if workers > rows:
        workers = rows
    return workers


def host_predict_task_count_work(rows: Int, work: Int) -> Int:
    """`host_predict_task_count(rows)` with the grain floor (lane
    perf-host-grain): `work` is the WHOLE call's scalar element operations
    (elements touched, times the inner length for a pairwise or reduction
    item). Below 2 * HOST_MIN_TASK_WORK the call is one task on the calling
    thread; above it, at most `work // HOST_MIN_TASK_WORK` tasks, so every
    task owns at least HOST_MIN_TASK_WORK operations. A `work` of 0 or less
    means unknown and applies no floor.

    ONLY for row-independent splits (every row a function of its own input
    and the fitted state, the contract of this module): the task count then
    moves no bit. A split whose partition feeds an ordered fold keeps its
    own partition and is not given this count."""
    var tasks = host_predict_task_count(rows)
    if tasks <= 1 or work <= 0:
        return tasks
    if work < 2 * HOST_MIN_TASK_WORK:
        return 1
    var cap = work // HOST_MIN_TASK_WORK
    if tasks > cap:
        tasks = cap
    return max(tasks, 1)


def host_grain_cap(tasks: Int, work: Int) -> Int:
    """A caller's row-task count `tasks` with the grain floor applied: 1
    when `work` (the whole call's scalar element operations) is below
    2 * HOST_MIN_TASK_WORK, else at most `work // HOST_MIN_TASK_WORK`;
    `work <= 0` (unknown) leaves `tasks` as given. For row-independent
    splits only, like `host_predict_task_count_work`."""
    if tasks <= 1 or work <= 0:
        return max(tasks, 1)
    if work < 2 * HOST_MIN_TASK_WORK:
        return 1
    return max(1, min(tasks, work // HOST_MIN_TASK_WORK))


@always_inline
def host_predict_chunk(rows: Int, tasks: Int) -> Int:
    """Rows per task, the ceiling of `rows / tasks`; task `c` owns
    `[c * chunk, min((c + 1) * chunk, rows))`."""
    if tasks < 1:
        return rows
    return (rows + tasks - 1) // tasks


#: The pointer every `_into` entry takes: the host bindings' own type
#: (`core/hostptr.mojo::f32_ptr`), so a binding hands the caller's
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
