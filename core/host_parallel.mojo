# SPDX-License-Identifier: Apache-2.0
"""THE ONE HOST THREAD SPLIT: `sync_parallelize` with a PINNED floating-point
environment (lane cpu, 2026-09-27, DEVIATION 5900).

EVERY host thread split in the tree goes through this module.
`tools/check_host_parallel_sites.py` (run by `pixi run check-host-parallel`)
refuses any `.mojo` file outside it that calls `sync_parallelize` directly.
This module also absorbs lane/algos-linear's `core/host_fp_env.mojo`
(`host_ieee_fp_enter` / `host_ieee_fp_leave` around each task body, main
0b7b6d5c1): `host_parallelize` below does the same job at the split rather
than inside each body, so that file and its calls in
`core/classical_host_predict.mojo` and `glm/estimator.mojo` were removed
when the two merged, and no second environment module is ever added.

MEASURED, NOT ASSUMED. On an x86 RunPod box (AMD EPYC 7352, the pinned
toolchain) the calling thread's MXCSR reads 0x1fa0 (the IEEE default) and
every worker thread `sync_parallelize` dispatches onto reads 0x9fe0: the
Mojo runtime's workers run with FTZ (bit 15) and DAZ (bit 6) set. A host
path therefore computed a subnormal result as a subnormal when its rows ran
on the calling thread and as a signed zero when they ran on a worker. The
bits depended on the THREAD COUNT and on which rows a task owned:
`LogisticRegression.predict_proba` (float64 `1 / (1 + exp(709.39))` =
8.22e-309) read DIVERGENT CPU against CUDA on the `dupes` fixture at the
default thread count and IDENTICAL at MOJOLEARN_CPU_THREADS=1, and the
GPU binding's serial host link agreed with the one-thread run.

THE MOVE IS PIN. There are exactly two entries:

- `host_parallelize(func, n)`: every task runs in the CALLING thread's
  environment. It reads MXCSR (x86) or FPCR (Arm) on the caller; each task
  installs it, runs the body and restores the worker's own value, so the
  runtime's pool is left as it was found. This is the entry for every host
  loop. The caller's environment is the one a one-task (serial) run uses,
  so a split's bits no longer depend on the task count.
- `host_parallelize_pool_env(func, n)`: the tasks keep the runtime WORKER's
  environment (FTZ+DAZ on x86). Only the GBDT fit's host regions use it
  (`gbdt/train.mojo`, `gbdt/resident_model.mojo`, `gbdt/host/gbdt_oracle.mojo`):
  every recorded GBDT column (CUDA, AMD, Apple and CPU) was computed with
  their border search and staging on pool workers, and pinning the caller's
  environment there moved 74 `denormal`-fixture cells of the gbdt,
  cross-val and saved-model lanes on the CUDA column (measured at 762f811cc).
  Moving GBDT to the caller's environment is a column re-record and is the
  trees lane's call (docs/lanes/progress/cpu.md, "the GBDT environment
  question"). Its bits do not depend on the task count: a pool task runs
  with FTZ+DAZ even at n = 1. OPEN, named: gbdt_oracle's serial small-fit
  arm (`n_rows * n_features < 2^18`) runs on the calling thread.

Float32 arithmetic spelled through `ftz` (`checks/numerics.mojo`, an integer
test) reads the same under either mode; what the pin fixes is everything
that is not: float64 host arithmetic (the probability links,
`portable_exp64`/`portable_log64`) and any float32 operation whose operand
or result can be subnormal before its flush. A target with neither register
(none today) runs the tasks as `sync_parallelize` does.

CELLS THAT MOVED (DEVIATION 5900; each was thread-count dependent before,
none moved at MOJOLEARN_CPU_THREADS=1): `logistic-unpenalized-no-intercept`
on `dupes` (infer, batch), CPU column only; it now equals the CUDA column.
The full per-lane record is in docs/lanes/progress/cpu.md.

`host_fp_env` / `host_fp_env_set` are exported for a caller that owns its
own threads.
"""
from std.sys import llvm_intrinsic
from std.sys.info import CompilationTarget
from std.memory import stack_allocation

from max.algorithm import sync_parallelize


@always_inline
def host_fp_env() -> UInt64:
    """The calling thread's floating-point control word: MXCSR on x86,
    FPCR on Arm, 0 elsewhere."""
    comptime if CompilationTarget.is_x86():
        var p = stack_allocation[1, UInt32]()
        p.store(0, UInt32(0))
        llvm_intrinsic["llvm.x86.sse.stmxcsr", NoneType](p)
        return UInt64(p.load(0))
    elif CompilationTarget.has_neon():
        return UInt64(llvm_intrinsic["llvm.aarch64.get.fpcr", Int64]())
    else:
        return UInt64(0)


@always_inline
def host_fp_env_set(v: UInt64):
    """Install a control word `host_fp_env` read (no-op elsewhere). The
    write is an intrinsic with side effects on memory, so the task's loads
    and the arithmetic that reads them stay after it."""
    comptime if CompilationTarget.is_x86():
        var p = stack_allocation[1, UInt32]()
        p.store(0, UInt32(v))
        llvm_intrinsic["llvm.x86.sse.ldmxcsr", NoneType](p)
    elif CompilationTarget.has_neon():
        llvm_intrinsic["llvm.aarch64.set.fpcr", NoneType](Int64(v))


def host_parallelize[FuncType: def(Int) -> None](ref func: FuncType, n: Int):
    """`sync_parallelize(func, n)`, every task in the caller's floating-point
    environment (DEVIATION 5900). The work split, the task count and the
    join are `sync_parallelize`'s, unchanged."""
    var env = host_fp_env()

    def task(i: Int) {imm func, imm env}:
        var saved = host_fp_env()
        host_fp_env_set(env)
        func(i)
        host_fp_env_set(saved)

    sync_parallelize(task, n)


def host_parallelize_pool_env[FuncType: def(Int) -> None](ref func: FuncType, n: Int):
    """`sync_parallelize(func, n)` with the runtime WORKER's environment
    (FTZ+DAZ on x86), for the GBDT fit's host regions only; see the module
    note. Every other host loop uses `host_parallelize`."""
    sync_parallelize(func, n)
