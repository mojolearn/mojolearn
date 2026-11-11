# SPDX-License-Identifier: Apache-2.0
"""The existing native host cluster entry, independent of Python envelopes."""
from core.hostptr import read_f32
from x_cluster.entries import run_entry
from x_cluster.host.host_ops import HostOps
from x_cluster.out import ClusterOut


def cluster_host_execute(which: Int, source: Int, count: Int, auxiliary: Int,
                         auxiliary_count: Int, ints: List[Int], floats: List[Float64]) raises -> ClusterOut:
    var x = read_f32(source, count) if count > 0 else List[Float32]()
    var a = read_f32(auxiliary, auxiliary_count) if auxiliary_count > 0 else List[Float32]()
    var ops = HostOps()
    var result = run_entry(ops, which, x, a, ints, floats)
    _ = ops^
    _ = x^
    _ = a^
    return result^
