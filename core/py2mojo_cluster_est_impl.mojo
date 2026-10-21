# SPDX-License-Identifier: Apache-2.0
"""Typed DBSCAN core-array loops moved word for word from bindings/py2mojo_cluster_est.mojo."""
from std.memory import memcpy


def dbscan_core_count(cp: MutPointer[UInt8, MutUntrackedOrigin], n: Int) raises -> Int:
    var n_core = 0
    for i in range(n):
        var f = cp.unsafe_load(i)
        if f > 1:
            raise Error("mojolearn DBSCAN: the fit's core mask holds a value other than 0 or 1")
        n_core += Int(f)
    return n_core


def dbscan_core_gather(
    xp: MutPointer[Float32, MutUntrackedOrigin], lp: MutPointer[Int32, MutUntrackedOrigin],
    cp: MutPointer[UInt8, MutUntrackedOrigin], n: Int, d: Int,
    ip: MutPointer[Int32, MutUntrackedOrigin], op: MutPointer[Float32, MutUntrackedOrigin],
    olp: MutPointer[Int32, MutUntrackedOrigin],
):
    var o = 0
    for i in range(n):
        if cp.unsafe_load(i) != 0:
            ip.unsafe_store(o, Int32(i))
            olp.unsafe_store(o, lp.unsafe_load(i))
            memcpy(dest=op + o * d, src=xp + i * d, count=d)
            o += 1
