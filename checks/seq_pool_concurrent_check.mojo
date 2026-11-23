# SPDX-License-Identifier: Apache-2.0
"""The sequence executor's process pool (sequence/exec_device.mojo `_SeqPool`)
under concurrent executors (lane r5-proving, 2026-11-23).

Two Python threads run device entries with the GIL released (the MLP door and
its model-selection member); each entry makes its own `DeviceExec`, and every
such executor leases device buffers and pinned stages from the one process
pool. Now each executor owns a whole sub-pool from construction to its end
(only the take and the give-back touch the registry, under its lock word), so
its leases are its own by construction. Before, with one shared pool and no
lock, two executors could lease the same free buffer or index a pool list
another thread was growing (L40S proving run
nvc1-0001: cross_val_score(MLPRegressor) from two threads, "Fatal Python
error: Illegal instruction").

Each worker runs ROUNDS rounds of: a new executor; buffers of the counts every
worker shares (so leases race for the same free entries) and one count of its
own (so the lists grow while other workers read them); its own values
uploaded, downloaded and compared; the executor ends (its leases go back).
A worker that reads another worker's values, or faults, fails the check.
Worker count: MOJOLEARN_NATIVE_TEST_WORKERS (the allocation), else the
logical cores, as checks/device_context_concurrent_repro.mojo.
"""
from max.algorithm import sync_parallelize
from std.sys.info import num_logical_cores
from std.os import getenv
from sequence.exec_device import DeviceExec
from sequence.ops import FP

comptime ROUNDS = 200


def _round(unit: Int, r: Int) raises:
    var ex = DeviceExec()
    var counts: List[Int] = [64, 1000, 4096, 4096 + 17 * (unit + 1) + r % 5]
    var hosts = List[List[Float32]]()
    var backs = List[List[Float32]]()
    var devs = List[Int]()
    for k in range(len(counts)):
        var n = counts[k]
        var host = List[Float32](length=n, fill=0.0)
        for i in range(n):
            host[i] = Float32(unit * 100000 + r * 10 + k) + Float32(i % 7)
        hosts.append(host^)
        backs.append(List[Float32](length=n, fill=-1.0))
        devs.append(Int(ex.alloc(n)))
    for k in range(len(counts)):
        ex.upload(FP(unsafe_from_address=devs[k]), FP(unsafe_from_address=Int(hosts[k].unsafe_ptr())), counts[k])
    for k in range(len(counts)):
        ex.download(FP(unsafe_from_address=Int(backs[k].unsafe_ptr())), FP(unsafe_from_address=devs[k]), counts[k])
    ex.sync()
    for k in range(len(counts)):
        for i in range(counts[k]):
            if backs[k][i] != hosts[k][i]:
                raise Error("worker " + String(unit) + " round " + String(r) + " buffer " + String(k)
                    + " read another executor's value")
    _ = hosts^
    _ = backs^
    _ = devs^
    _ = ex^


def main() raises:
    var configured = getenv("MOJOLEARN_NATIVE_TEST_WORKERS")
    var workers = num_logical_cores() if configured == "" else Int(atol(configured))
    if workers < 2:
        raise Error("at least two allocated workers required")
    var failures = List[Int](length=workers, fill=0)
    var fp = failures.unsafe_ptr()

    def task(unit: Int) {imm fp}:
        try:
            for r in range(ROUNDS):
                _round(unit, r)
        except error:
            print("FAIL worker", unit, String(error))
            fp.unsafe_store(unit, 1)

    print("START sequence pool, concurrent executors; workers", workers, "rounds", ROUNDS)
    sync_parallelize(task, workers)
    var failed = 0
    for value in failures:
        failed += value
    _ = failures^
    if failed:
        raise Error("one or more concurrent executors failed")
    print("PASS sequence pool, concurrent executors; workers", workers)
