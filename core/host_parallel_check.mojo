# SPDX-License-Identifier: Apache-2.0
"""`pixi run check-host-parallel`: DEVIATION 5900's seam, host worker
threads compute in the caller's floating-point environment.

The oracle is the calling thread: `1.0 / 1.2e308` in float64 is the
subnormal 8.333333333333336e-309 there (the IEEE default). Every task of
`host_parallelize` must produce those bits, at every task count, and read
the caller's control word.

SEPARATION FIRST. The same division under plain `sync_parallelize` must
DIFFER from the oracle on at least one task (the runtime's workers flush),
or the fixture cannot tell the pinned spelling from the unpinned one and
the check refuses as VACUOUS. On a platform whose workers already run in
the caller's mode it is VACUOUS by construction, and says so; that is a
reach failure of this fixture there, never a pass.
"""
from std.os import abort
from std.memory import bitcast

from max.algorithm import sync_parallelize

from core.host_parallel import (
    host_fp_env,
    host_parallelize,
    host_parallelize_pool_env,
)


def _bits(x: Float64) -> UInt64:
    return bitcast[DType.uint64](x)


def main() raises:
    var a = Float64(1.0)
    var b = Float64(1.2e308)
    var oracle = a / b
    if _bits(oracle) == UInt64(0):
        abort("host_parallel_check: the calling thread flushes; no oracle")
    var env = host_fp_env()
    comptime N = 16
    var raw = List[Float64](length=N, fill=Float64(-1.0))
    var rp = raw.unsafe_ptr()

    def raw_task(i: Int) {imm rp, imm a, imm b}:
        rp.store(i, a / b)

    sync_parallelize(raw_task, N)
    var separates = False
    for i in range(N):
        if _bits(raw[i]) != _bits(oracle):
            separates = True
    if not separates:
        print("host_parallel_check: VACUOUS: plain sync_parallelize already "
              + "computes in the caller's mode on this platform")
        return
    var bad = 0
    # The GBDT entry keeps the WORKER's environment: its bits are the plain
    # split's, task for task (the recorded GBDT columns; see the module).
    var pool = List[Float64](length=N, fill=Float64(-1.0))
    var pp = pool.unsafe_ptr()

    def pool_task(i: Int) {imm pp, imm a, imm b}:
        pp.store(i, a / b)

    host_parallelize_pool_env(pool_task, N)
    for i in range(N):
        if _bits(pool[i]) != _bits(raw[i]):
            print("  pool entry task", i, "result", pool[i], "!= plain worker", raw[i])
            bad += 1
    for tasks in [1, 2, 3, 7, 16]:
        var got = List[Float64](length=N, fill=Float64(-1.0))
        var envs = List[UInt64](length=N, fill=UInt64(0))
        var gp = got.unsafe_ptr()
        var ep = envs.unsafe_ptr()

        def task(i: Int) {imm gp, imm ep, imm a, imm b}:
            gp.store(i, a / b)
            ep.store(i, host_fp_env())

        host_parallelize(task, tasks)
        for i in range(tasks):
            if _bits(got[i]) != _bits(oracle):
                print("  tasks", tasks, "task", i, "result", got[i], "!= oracle", oracle)
                bad += 1
            if envs[i] & ~UInt64(0x3F) != env & ~UInt64(0x3F):
                # the low six MXCSR bits are sticky exception FLAGS, not modes
                print("  tasks", tasks, "task", i, "env", hex(envs[i]), "!= caller", hex(env))
                bad += 1
    if bad:
        abort("host_parallel_check: FAIL (DEVIATION 5900): " + String(bad)
              + " task(s) did not compute in the caller's environment")
    print("host_parallel_check: PASS (caller env " + hex(env)
          + "; plain workers separate, host_parallelize matches the oracle at 1, 2, 3, 7, 16 tasks,"
          + " host_parallelize_pool_env matches the plain workers)")
