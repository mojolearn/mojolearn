# SPDX-License-Identifier: Apache-2.0
"""THE ONE HOST THREAD SPLIT: `sync_parallelize` with the caller's
floating-point environment (lane cpu, 2026-09-27, DEVIATION 5900).

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

THE MOVE IS PIN: every task runs in the calling thread's environment.
`host_parallelize` reads MXCSR (x86) or FPCR (Arm) on the calling thread,
and each task installs it, runs the body, and restores the worker's own
value, so the runtime's pool is left as it was found. Float32 arithmetic
spelled through `ftz` (`checks/numerics.mojo`, an integer test) reads the
same under either mode; what this pins is everything that is not: float64
host arithmetic (the probability links, `portable_exp64`/`portable_log64`)
and any float32 operation whose operand or result can be subnormal before
its flush. A target with neither register (none today) runs the tasks as
`sync_parallelize` does.

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
    """Install a control word `host_fp_env` read (no-op elsewhere)."""
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
